import copy
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from domainbed.dataloaders import dataloader_factory
from domainbed.lib import swa_utils
from domainbed import swad
from domainbed.algorithms import BAIR

from torch import einsum
from einops import rearrange
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ot
import math
import pdb


		
def set_tr_val_samples_labels(meta_filenames, val_size):
	sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels = [], [], [], []

	for idx_domain, meta_filename in enumerate(meta_filenames):
		column_names = ["filename", "class_label"]
		data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
		data_frame = data_frame.sample(frac=1).reset_index(drop=True)

		split_idx = int(len(data_frame) * (1 - val_size))
		sample_tr_paths.append(data_frame["filename"][:split_idx])
		class_tr_labels.append(data_frame["class_label"][:split_idx])

		sample_val_paths.extend(data_frame["filename"][split_idx:])
		class_val_labels.extend(data_frame["class_label"][split_idx:])
	return sample_tr_paths, class_tr_labels, sample_val_paths, class_val_labels


def set_test_samples_labels(meta_filenames):
	sample_paths, class_labels = [], []
	for idx_domain, meta_filename in enumerate(meta_filenames):
		column_names = ["filename", "class_label"]
		data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
		sample_paths.extend(data_frame["filename"])
		class_labels.extend(data_frame["class_label"])
	return sample_paths, class_labels



class Trainer:
	def __init__(self, hparams, dataset_configs, device, args):
		self.dataset_configs = dataset_configs
		self.device = device
		self.args = args
		# Read data list files and split Train-Val with 80% train, 20% test
		(
			src_tr_sample_paths,
			src_tr_class_labels,
			src_val_sample_paths,
			src_val_class_labels,
		) = set_tr_val_samples_labels(self.dataset_configs.src_train_meta_filenames, self.dataset_configs.val_size)
		test_sample_paths, test_class_labels = set_test_samples_labels(self.dataset_configs.target_test_meta_filenames)
		
		self.train_loaders = []
		
		# Create train dataloader
		for i in range(self.dataset_configs.n_domain_classes):
			self.train_loaders.append(
				DataLoader(
					dataloader_factory.get_train_dataloader(self.dataset_configs.dataset)(
						src_path=self.dataset_configs.src_data_path,
						sample_paths=src_tr_sample_paths[i],
						class_labels=src_tr_class_labels[i],
						domain_label=i,
					),
					batch_size=self.dataset_configs.batch_size,
					shuffle=True,
					drop_last=True, num_workers=2
				)
			)

		# Create val dataloader

		self.val_loader = DataLoader(
			dataloader_factory.get_test_dataloader(self.dataset_configs.dataset)(
				src_path=self.dataset_configs.src_data_path,
				sample_paths=src_val_sample_paths,
				class_labels=src_val_class_labels,
			),
			batch_size=self.dataset_configs.batch_size,
			shuffle=False, num_workers=2
		)

		# Create test dataloader
		self.test_loader = DataLoader(
			dataloader_factory.get_test_dataloader(self.dataset_configs.dataset)(
				src_path=self.dataset_configs.src_data_path, sample_paths=test_sample_paths, class_labels=test_class_labels
			),
			batch_size=self.dataset_configs.batch_size,
			shuffle=False, num_workers=2
		)
		# Log number of images 
		for i in range(self.dataset_configs.n_domain_classes):
			print('Train: ', i, ' ', len(self.train_loaders[i].dataset))

		print('Val_size: ', self.dataset_configs.val_size)
		print('Val: ', len(self.val_loader.dataset))
		print('Test: ', len(self.test_loader.dataset))

		self.algorithm = BAIR(
			(3, 224, 224),
			self.dataset_configs.n_classes,
			self.dataset_configs.n_domain_classes,
			hparams).to(self.device)
		
		self.num_embed = self.dataset_configs.n_classes * hparams['prototype_per_class']

		
		# Define SWAD average model
		self.swad_algorithm = swa_utils.AveragedModel(self.algorithm.network)
		self.swad_valley = swad.LossValley(evaluator=None, n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
		self.hist = None

		self.criterion = nn.CrossEntropyLoss()

		self.ret = {}
		
		self.val_loss_min = np.Inf
		self.test_acc_max = 0
		self.val_acc_max = 0
		self.corresponding_test = 0


	def train(self):
		self.algorithm.train()

		self.train_iter_loaders = []
		for train_loader in self.train_loaders:
			self.train_iter_loaders.append(iter(train_loader))

		for iteration in range(self.dataset_configs.iterations):
			samples, labels, domain_labels = [], [], []

			for idx in range(len(self.train_iter_loaders)):
				# Reset Loader
				if (iteration % (len(self.train_iter_loaders[idx]))-1) == 0:
					self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
				
				# Load Mini-Batch
				itr_samples, itr_labels, itr_domain_labels = next(self.train_iter_loaders[idx])
				samples.append(itr_samples.to(self.device))
				labels.append(itr_labels.to(self.device))

			# import pdb; pdb.set_trace()

			batches = {'x': samples, 'y': labels}
			inputs = {**batches, "step": iteration}
			step_vals = self.algorithm.update(**inputs)
			
			if iteration > self.dataset_configs.iterations * self.args.start_swad:
				# Update swad average model
				self.swad_algorithm.update_parameters(self.algorithm.network, step=iteration)

				if iteration % self.dataset_configs.step_eval == 0 or iteration == self.dataset_configs.iterations - 1:
					val_acc, val_loss = self.evaluate(iteration)

					if iteration > self.dataset_configs.iterations * self.args.start_swad:
						self.swad_valley.update_and_evaluate(self.swad_algorithm, val_acc, val_loss)
						if self.swad_valley.dead_valley:
							break
						self.swad_algorithm = swa_utils.AveragedModel(self.algorithm.network)
		

		self.algorithm.eval()
		final_swad = self.swad_valley.get_final_model().module
		if not isinstance(final_swad, nn.Sequential):
			final_swad = final_swad.module
		test_acc, _ = self.evaluate_loader(final_swad, test_type='Test', model_type='swad')
		self.ret["classifer weight"] = test_acc

		self.histogram()
		self.save_model(final_swad, 'swad')
		self.save_model(self.algorithm.network, 'last')
		return self.ret

	def histogram(self, data=None, labels=None):
	   
		self.hist = torch.zeros(len(self.train_loaders), self.num_embed).to(self.device)
		fault_map = 0
		prototype_feature = self.algorithm.prototype_net.prototype
		for idx in range(len(self.train_loaders)):
			for iteration, (samples, labels, domains) in enumerate(self.train_loaders[idx]):
				samples = samples.to(self.device)
				labels = labels.to(self.device)
				domains = domains.to(self.device)
				features = self.algorithm.network[0](samples) 
				cost_matrix = self.algorithm.cosine_similarity(features, prototype_feature)
				sub_space_idx = cost_matrix.min(1)[1]
				fault_map = (1-(self.algorithm.prototype_net.labels[sub_space_idx] == labels).int()).sum()
				for v in sub_space_idx:
					self.hist[idx, v]+=1
		
		print(self.hist.int())
		print(self.hist.sum(0).int())
		print('fault_map:' , fault_map)

	def evaluate_loader(self, model, test_type='Val', model_type='norm'):
		n_class_corrected = 0
		total_classification_loss = 0
		swad_norm_ave_prototype_n_class_corrected = 0
		if test_type == 'Val':
			loader = self.val_loader
		else:
			loader = self.test_loader
		
	   
		with torch.no_grad():
					
			# Using prototype
			prototype_logits = self.algorithm.network[1](model[2].prototype)
			_, prototype_predicted_classes = torch.max(prototype_logits, 1)
			average_prototype = torch.zeros(self.dataset_configs.n_classes, self.dataset_configs.feature_dim).to(self.device)
			norm_average_prototype = torch.zeros(self.dataset_configs.n_classes, self.dataset_configs.feature_dim).to(self.device)
			for i in range(self.dataset_configs.n_classes):
				average_prototype[i]=model[2].prototype[prototype_predicted_classes==i].mean(0)
				norm_average_prototype[i]=F.normalize(model[2].prototype[prototype_predicted_classes==i],  dim=1).mean(0)

			prototype_logits = self.algorithm.network[1](self.algorithm.prototype_net.prototype)
			_, prototype_predicted_classes = torch.max(prototype_logits, 1)
			average_last_prototype = torch.zeros(self.dataset_configs.n_classes, self.dataset_configs.feature_dim).to(self.device)
			norm_average_last_prototype = torch.zeros(self.dataset_configs.n_classes, self.dataset_configs.feature_dim).to(self.device)
			
			for i in range(self.dataset_configs.n_classes):
				average_last_prototype[i]=self.algorithm.prototype_net.prototype[prototype_predicted_classes==i].mean(0)
				norm_average_last_prototype[i]=F.normalize(self.algorithm.prototype_net.prototype[prototype_predicted_classes==i],  dim=1).mean(0)
				

			for iteration, (samples, labels, domain_labels) in enumerate(loader):
				samples, labels = samples.to(self.device), labels.to(self.device)
				
				features = model[0](samples)
				
				# Using classifier for prediction
				predicted_classes = model[1](features)
				classification_loss = self.criterion(predicted_classes, labels)
				total_classification_loss += classification_loss.item()
				_, predicted_classes = torch.max(predicted_classes, 1)
				n_class_corrected += (predicted_classes == labels).sum().item()
				
				# Using swad-encoder with last prototype istead of swad-prototype
				if model_type == 'swad':
					predicted_classes = F.linear(F.normalize(features, dim=1), F.normalize(average_last_prototype,  dim=1))
					_, predicted_classes = torch.max(predicted_classes, 1)
					swad_norm_ave_prototype_n_class_corrected += (predicted_classes == labels).sum().item()
					
		
		print("-----------------------------------")
		print_out = "{} set: Accuracy: {}/{} {:.2f}%, {}".format(
			test_type, 
			n_class_corrected,len(loader.dataset),
			100.0 * n_class_corrected / len(loader.dataset),
			total_classification_loss / len(loader.dataset),)
		print(print_out)
		print("-----------------------------------")
		
		if model_type == 'swad':
			print("-----------------------------------")
			print_out_swad = "{} set: SWAD-encoder + Last-protype Accuracy: {}/{} {:.2f}%, {}".format(
				test_type, 
				n_class_corrected,len(loader.dataset),
				100.0 * swad_norm_ave_prototype_n_class_corrected / len(loader.dataset), 
				total_classification_loss / len(loader.dataset),)
			print(print_out_swad)
			print("-----------------------------------")
			self.ret["prototype"] = swad_norm_ave_prototype_n_class_corrected / len(loader.dataset)

		return n_class_corrected / len(loader.dataset), total_classification_loss / len(loader.dataset)
		

	def evaluate(self, n_iter):
		self.algorithm.eval()
		
		val_acc, val_loss = self.evaluate_loader(self.algorithm.network, test_type='Val')
		test_acc, _ = self.evaluate_loader(self.algorithm.network, test_type='Test')
	 
		if self.val_acc_max < val_acc:
			self.val_acc_max = val_acc
			self.corresponding_test = test_acc
			self.save_model(self.algorithm.network, 'best_val')

			
		if self.test_acc_max < test_acc:
			self.test_acc_max = test_acc
			self.save_model(self.algorithm.network, 'best_test')


		print( "Best val: {:.2f}%, Corres: {:.2f}%, Best Test: {:.2f}%".format(
			100.0 * self.val_acc_max, 100.0 * self.corresponding_test, 100.0 * self.test_acc_max))
		
		self.algorithm.train()
		return val_acc, val_loss

	def test(self):
		self.algorithm.network.eval()
		val_acc, _ = self.evaluate_loader(self.algorithm.network, test_type='Val')
		test_acc, _ = self.evaluate_loader(self.algorithm.network, test_type='Test')
		print("val: {}, Test: {}".format(val_acc, test_acc))


	def save_model(self, network, name='best_model'):
		if self.args.save_model_dir == '':
			return
		name = '{}_{}_{}_{}'.format(name, 
				self.args.dataset, 
				str(self.args.target), 
				str(self.args.seed))
		
		torch.save({
			'encoder_state_dict': network[0].state_dict(), 
			'classifier_state_dict': network[1].state_dict(), 
			'prototype_state_dict': network[2].state_dict()}, 
			os.path.join(self.args.save_model_dir, f'{name}.pth'))


	def load_model(self, ckpt):
		state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage)
		encoder_state = state_dict["encoder_state_dict"]
		classifier_state = state_dict["classifier_state_dict"]
		prototype_state = state_dict["prototype_state_dict"]
		self.algorithm.featurizer.load_state_dict(encoder_state)
		self.algorithm.classifier.load_state_dict(classifier_state)
		self.algorithm.prototype_net.load_state_dict(prototype_state)


	def save_plot(self, plot_dir, name_folder=None):
		self.network.eval()

		feature_train, Y_train, Y_domain_train = [], [], []
		feature_test, Y_test, Y_domain_test = [], [], []
		mask_test = []

		self.train_iter_loaders = []
		with torch.no_grad():
			for train_loader in self.train_loaders:

				for iteration, (samples, labels, domain_labels) in enumerate(train_loader):
					samples = samples.to(self.device)
					labels = labels.to(self.device)
					domain_labels = domain_labels.to(self.device)

					features = self.algorithm.network[0](samples)
					
					feature_train += features.tolist()
					Y_train += labels.tolist()
					Y_domain_train += domain_labels.tolist()
			print("Train dumped")
			
			for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
				samples, labels = samples.to(self.device), labels.to(self.device)
				features = self.algorithm.network[0](samples)
				feature_test += features.tolist()
				Y_test += labels.tolist()
				Y_domain_test += domain_labels.tolist()
			print("Test dumped")

		if not os.path.exists(plot_dir):
			os.mkdir(plot_dir)
		if name_folder is None:
			name_folder = "{}_{}_seed_{}".format(self.args.dataset, self.args.target, self.args.seed)
		fol_name = os.path.join(plot_dir, name_folder)
		os.makedirs(fol_name, exist_ok=True)
		print('Save at', fol_name)
		with open(os.path.join(fol_name, "prototype.pkl"), "wb") as fp:
			pickle.dump(self.network[2].prototype.tolist(), fp)

		with open(os.path.join(fol_name, "class_weight.pkl"), "wb") as fp:
			pickle.dump(self.network[1].classifier.weight.tolist(), fp)

		with open(os.path.join(fol_name, "feature_train.pkl"), "wb") as fp:
			pickle.dump(feature_train, fp)
		with open(os.path.join(fol_name, "Y_train.pkl"), "wb") as fp:
			pickle.dump(Y_train, fp)
		with open(os.path.join(fol_name, "Y_domain_train.pkl"), "wb") as fp:
			pickle.dump(Y_domain_train, fp)

		with open(os.path.join(fol_name, "feature_test.pkl"), "wb") as fp:
			pickle.dump(feature_test, fp)
		with open(os.path.join(fol_name, "Y_test.pkl"), "wb") as fp:
			pickle.dump(Y_test, fp)
		with open(os.path.join(fol_name, "Y_domain_test.pkl"), "wb") as fp:
			pickle.dump(Y_domain_test, fp)
