import copy
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from BAIR.src.dataloaders import dataloader_factory
from BAIR.src.models import model_factory
from BAIR.src.models.kernel import MLP
from BAIR.src.models import swa_utils
from BAIR.src.models import swad
from BAIR.src.models.loss import AdversarialNetwork, calc_coeff, Entropy, grl_hook
from BAIR.src.models.res_decoder import Decoder


from torch import einsum
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import ot
import math
import torch 
import torch.autograd as autograd
import json
import pdb


class Classifier(nn.Module):
	def __init__(self, feature_dim, n_classes, is_bias):
		super(Classifier, self).__init__()
		if is_bias == 1:
			# import pdb; pdb.set_trace()
			self.classifier = nn.Linear(int(feature_dim), n_classes, bias=True)
		else:
			# import pdb; pdb.set_trace()
			self.classifier = nn.Linear(int(feature_dim), n_classes, bias=False)

	def forward(self, x):
		y = self.classifier(x)
		return y


class GradReverse(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.save_for_backward(-alpha)
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		alpha = ctx.saved_tensors[0]
		if ctx.needs_input_grad[0]:
			grad_output = (grad_output * (alpha))
		return (grad_output, None)


class Identity(nn.Module):
	"""An identity layer"""

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Prototype(nn.Module):
	def __init__(self, num_embed, n_classes, n_domains, prototype_per_class, feature_dim=2048):
		super(Prototype, self).__init__()
		self.n_classes = n_classes
		self.n_domains = n_domains
		self.num_embed = num_embed
		self.weight = nn.Parameter(torch.log(torch.ones(n_domains, num_embed)/num_embed))
		self.prototype = nn.Parameter(torch.ones(num_embed, feature_dim))
		self.prototype.data.uniform_(-1.0 /  num_embed, 1.0 /  num_embed)
		self.register_buffer('labels', torch.tensor(range(n_classes)).unsqueeze(-1).repeat(1, prototype_per_class).reshape(-1))

	def forward(self, batch_classes, d_idx=0):
		mask = (self.labels.unsqueeze(1)==batch_classes.unsqueeze(0)).sum(1) > 0
		selected_prototypes = self.prototype[torch.where(mask > 0)]
		selected_prototypes_labels = self.labels[torch.where(mask > 0)]
		selected_prototype_weight = self.weight[d_idx][torch.where(mask > 0)]
		return selected_prototypes, selected_prototype_weight, selected_prototypes_labels
	
	def update_label(self, predicted_classes):
		self.labels = predicted_classes

	def get_orthogonalization_loss(self):
		r_1 = torch.sqrt(torch.sum(self.prototype**2,dim=1,keepdim=True))
		topic_metrix = torch.mm(self.prototype, self.prototype.T.float()) / torch.mm(r_1, r_1.T)
		topic_metrix = torch.clamp(topic_metrix.abs(), 0, 1)

		l1 = torch.sum(topic_metrix.abs())
		l2 = torch.sum(topic_metrix ** 2)

		loss_sparse = l1 / l2
		# loss_constraint = torch.abs(l1 - topic_metrix.shape[0]) / topic_metrix.shape[0]
		loss_constraint = torch.abs(l1 - topic_metrix.shape[0])

		# # better
		# r_loss = loss_sparse + 0.5*loss_constraint + 0.5*torch.sum((topic_metrix - torch.eye(model.topic.shape[0]).cuda())**2)
		# make sense
		# r_loss = loss_sparse + 0.5*loss_constraint + 0.5*torch.sum((torch.mm(model.topic.float(), model.topic.T.float()) - torch.eye(model.topic.shape[0]).cuda())**2)
		# import pdb; pdb.set_trace()
		return (loss_sparse + 0.5 * loss_constraint) / (topic_metrix.shape[0]**2)


class MaxInfo(nn.Module):

	def __init__(self, num_classes, scale=0.07):
		super(MaxInfo, self).__init__()
		self.soft_plus = nn.Softplus()
		self.label = torch.LongTensor([i for i in range(num_classes)])
		self.scale = 1 / scale
	
	def forward(self, feature, target, proxy):
		pred = F.linear(feature, proxy)  # (N, C)
		label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
		pred = torch.masked_select(pred.transpose(1, 0), label)  # N,
		
		pred = pred.unsqueeze(1)  # (N, 1)
		
		feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
		feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)
		
		logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
		label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
		loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
		
		return loss

		
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
	def __init__(self, args, device, bash_args):
		self.args = args
		self.device = device
		self.bash_args = bash_args
		# Read data list files and split Train-Val with 80% train, 20% test
		(
			src_tr_sample_paths,
			src_tr_class_labels,
			src_val_sample_paths,
			src_val_class_labels,
		) = set_tr_val_samples_labels(self.args.src_train_meta_filenames, self.args.val_size)
		test_sample_paths, test_class_labels = set_test_samples_labels(self.args.target_test_meta_filenames)
		
		self.train_loaders = []
		
		# Create train dataloader
		for i in range(self.args.n_domain_classes):
			self.train_loaders.append(
				DataLoader(
					dataloader_factory.get_train_dataloader(self.args.dataset)(
						src_path=self.args.src_data_path,
						sample_paths=src_tr_sample_paths[i],
						class_labels=src_tr_class_labels[i],
						domain_label=i,
					),
					batch_size=self.args.batch_size,
					shuffle=True,
					drop_last=True, num_workers=2
				)
			)

		# Create val dataloader

		self.val_loader = DataLoader(
			dataloader_factory.get_test_dataloader(self.args.dataset)(
				src_path=self.args.src_data_path,
				sample_paths=src_val_sample_paths,
				class_labels=src_val_class_labels,
			),
			batch_size=self.args.batch_size,
			shuffle=False, num_workers=2
		)

		# Create test dataloader
		self.test_loader = DataLoader(
			dataloader_factory.get_test_dataloader(self.args.dataset)(
				src_path=self.args.src_data_path, sample_paths=test_sample_paths, class_labels=test_class_labels
			),
			batch_size=self.args.batch_size,
			shuffle=False, num_workers=2
		)

		# Log number of images 
		for i in range(self.args.n_domain_classes):
			print('Train: ', i, ' ', len(self.train_loaders[i].dataset))
		print('Val_size: ', self.args.val_size)
		print('Val: ', len(self.val_loader.dataset))
		print('Test: ', len(self.test_loader.dataset))
		

		# Define Base Network
		self.encoder = model_factory.get_model(self.args.model)(dropout_rate=self.args.dropout_rate).to(self.device)
		
		self.classifier = Classifier(feature_dim=self.args.feature_dim, 
									 n_classes=self.args.n_classes, is_bias=self.bash_args.is_bias).to(self.device)
		self.env_classifier = Classifier(feature_dim=self.args.feature_dim+1, 
									 n_classes=self.args.n_classes, is_bias=self.bash_args.is_bias).to(self.device)

		
		# Define Prototypes
		self.num_embed = self.args.n_classes * self.bash_args.prototype_per_class
		self.prototype_net = Prototype(self.num_embed, self.args.n_classes, self.args.n_domain_classes, self.bash_args.prototype_per_class, self.args.feature_dim).to(self.device)

		self.domain_indx = [torch.full((self.args.batch_size, 1), indx).to(self.device) for indx in range(self.args.n_domain_classes)]
		self.scale = torch.tensor(-self.bash_args.ot_scale, requires_grad=False)
		
		# Define Disciminator for Prototype_DANN
		hparams = {}
		hparams["mlp_width"] = 256
		hparams["mlp_depth"] = 3
		hparams["mlp_dropout"] = 0.5
		self.alpha = torch.tensor(self.bash_args.disc_weight, requires_grad=False)
		self.discriminator = MLP(self.args.feature_dim, self.args.n_domain_classes, hparams).to(device)
		self.subspace_embeddings = nn.Embedding(self.num_embed, self.args.feature_dim).to(device)
		self.domain_embeddings = nn.Embedding(self.args.n_domain_classes, self.args.feature_dim).to(device)
		
		self.network = nn.Sequential(self.encoder, self.classifier, self.prototype_net)        
		
		self.optimizer = torch.optim.Adam(
			list(self.prototype_net.parameters())+
			list(self.encoder.parameters())+
			list(self.env_classifier.parameters())+
			list(self.classifier.parameters())+
			list(self.domain_embeddings.parameters()),
			lr=self.args.learning_rate, weight_decay=0.0)
		
		self.disc_optimizer = torch.optim.Adam(
			list(self.subspace_embeddings.parameters())+
			list(self.discriminator.parameters()), 
			lr=self.args.learning_rate, 
			weight_decay=0.0, betas=(0.5, 0.9))
		
		
		# Define SWAD average model
		self.swad_algorithm = swa_utils.AveragedModel(self.network)
		self.swad_valley = swad.LossValley(evaluator=None, n_converge=3, n_tolerance=6, tolerance_ratio=0.3)
		self.hist = None

		# Define loss function
		self.criterion = nn.CrossEntropyLoss()
		self.maxinfo_loss = MaxInfo(num_classes=self.args.n_classes)

		self.soft_criterion = nn.BCEWithLogitsLoss()

		self.val_loss_min = np.Inf
		self.test_acc_max = 0
		self.val_acc_max = 0
		self.corresponding_test = 0
		

	# METRIC for WS ----------------------------------------------
	def cosine_similarity(self, domain_features, prototype_feature):
		normed_domain_features = F.normalize(domain_features, dim=1)
		normed_prototype = F.normalize(prototype_feature, dim=1)
		pair_wises = torch.einsum('bd,dn->bn', normed_domain_features, rearrange(normed_prototype, 'n d -> d n'))
		similarity = 1 - pair_wises
		return similarity


	def set_writer(self, log_dir):
		if not os.path.exists(log_dir):
			os.mkdir(log_dir)
		shutil.rmtree(log_dir)
		return SummaryWriter(log_dir)


	def train(self):
		self.network.train()
		self.subspace_embeddings.train()
		self.discriminator.train()
		self.env_classifier.train()
		self.domain_embeddings.train()


		n_domain_corrected = 0
		n_class_corrected = 0
		prototype_n_class_corrected = 0
		total_samples = 0
		prototype_total_samples = 0

		self.train_iter_loaders = []
		for train_loader in self.train_loaders:
			self.train_iter_loaders.append(iter(train_loader))
		
		for iteration in range(self.args.iterations):
			samples, labels, domain_labels = [], [], []

			for idx in range(len(self.train_iter_loaders)):
				# Reset Loader
				if (iteration % (len(self.train_iter_loaders[idx]))-1) == 0:
					self.train_iter_loaders[idx] = iter(self.train_loaders[idx])
				
				# Load Mini-Batch
				itr_samples, itr_labels, itr_domain_labels = next(self.train_iter_loaders[idx])
				samples.append(itr_samples)
				labels.append(itr_labels)
				domain_labels.append(itr_domain_labels)

			tr_samples = torch.cat(samples, dim=0).to(self.device)
			tr_labels = torch.cat(labels, dim=0).to(self.device)
			tr_domain_labels = torch.cat(domain_labels, dim=0).to(self.device)

			total_loss = 0
			total_smooth_loss = 0

			# Classification Loss
			features = self.network[0](tr_samples)
			logits = self.network[1](features)
			cls_loss = self.criterion(logits, tr_labels)
			total_loss += cls_loss
			# # Enforcing Invariant Prediction Loss
			self.domain_indx = [torch.full((self.args.batch_size, 1), indx).to(self.device) for indx in range(self.args.n_domain_classes)]
			embeddings = torch.cat([curr_dom_embed for curr_dom_embed in self.domain_indx]).to(self.device)
			env_logits = self.env_classifier(torch.cat([features, embeddings], 1))

			# env_logits = self.env_classifier(features + self.domain_embeddings(tr_domain_labels))
			env_loss = self.criterion(env_logits, tr_labels)
			inv_loss =  self.bash_args.env_weight * (cls_loss - env_loss) ** 2 
			total_loss += env_loss + inv_loss

				
			# Log Classification Loss
			total_samples += features.shape[0]
			_, predicted_classes = torch.max(logits, 1)
			n_class_corrected += (predicted_classes == tr_labels).sum().item()

			# Test whether prototype can be used to make predition
			prototype_logits = self.network[1](self.prototype_net.prototype)
			_, prototype_predicted_classes = torch.max(prototype_logits, 1)


			prototype_total_samples += self.prototype_net.prototype.shape[0]
			prototype_n_class_corrected += (prototype_predicted_classes == self.prototype_net.labels).sum().item()
			

			if iteration > self.args.step_eval:
				self.prototype_net.update_label(prototype_predicted_classes)

			predicted_prototype = F.linear(features, self.prototype_net.prototype)
			softmax_prototype = nn.Softmax(dim=1)(predicted_prototype)
			
			# Feature regularization via contrastive learning -> maximun I(g(X),X) for each source domains

			if self.bash_args.pcl_weight > 0:
				for d_index in range(self.args.n_domain_classes): 
					normed_domain_features = F.normalize(features[self.args.batch_size*d_index:self.args.batch_size*(d_index+1),:], dim=1)
					domain_class_labels = tr_labels[self.args.batch_size*d_index:self.args.batch_size*(d_index+1)]
					loss_info = self.maxinfo_loss(normed_domain_features, domain_class_labels, F.normalize(self.classifier.classifier.weight, dim=1))
					total_loss += self.bash_args.pcl_weight * loss_info / self.args.n_domain_classes
				
			# Sub-space projetion via Wasserstein with different metric
			for d_index in range(self.args.n_domain_classes): 
				domain_features = features[self.args.batch_size*d_index:self.args.batch_size*(d_index+1),:]
				prototype_feature = self.prototype_net.prototype
				
				sample_weight = torch.ones(domain_features.shape[0]).to(self.device) / domain_features.shape[0]
				prototype_weight = torch.ones(prototype_feature.shape[0]).to(self.device) / prototype_feature.shape[0]
				cost_matrix = self.cosine_similarity(GradReverse.apply(domain_features, self.scale), prototype_feature)
				
				ot_cost = ot.emd2(sample_weight, prototype_weight.detach(), cost_matrix, numItermax=500000, return_matrix=True)
				total_loss += self.bash_args.ot_weight * ot_cost[0]
				# import pdb; pdb.set_trace()

				# sub-space balanced alignment via Prototype-DANN
				if self.bash_args.disc_weight > 0:
					softmax_output = softmax_prototype[self.args.batch_size*d_index:self.args.batch_size*(d_index+1)]
					entropy = Entropy(softmax_output)
					entropy.register_hook(grl_hook(self.bash_args.disc_weight))
					entropy = 1.0+torch.exp(-entropy)
					weight = entropy / torch.sum(entropy).detach().item()

					sub_space_idx = cost_matrix.min(1)[1]
					disc_input = GradReverse.apply(domain_features, self.alpha) + self.subspace_embeddings(sub_space_idx)
					
					domain_logit =  self.discriminator(disc_input)
					domain_loss = F.cross_entropy(domain_logit, tr_domain_labels[self.args.batch_size*d_index:self.args.batch_size*(d_index+1)], reduction='none')
					domain_loss = (weight * domain_loss).sum()

					if domain_loss.mean().item() > self.bash_args.disc_threshold or iteration < self.args.step_eval:
						domain_logit =  self.discriminator(disc_input.detach())
						domain_loss = F.cross_entropy(domain_logit, tr_domain_labels[self.args.batch_size*d_index:self.args.batch_size*(d_index+1)], reduction='none')
						domain_loss = (weight.detach() * domain_loss).sum()

					total_loss += domain_loss / self.args.n_domain_classes
					_, predicted_domain = torch.max(domain_logit, 1)
					n_domain_corrected += (predicted_domain == tr_domain_labels[self.args.batch_size*d_index:self.args.batch_size*(d_index+1)]).sum().item()

			
			if total_loss != 0:
				self.optimizer.zero_grad()
				self.disc_optimizer.zero_grad()
				total_loss.backward()
				self.optimizer.step()
				self.disc_optimizer.step()        

			if iteration > self.args.iterations * self.bash_args.start_swad:
				# Update swad average model
				self.swad_algorithm.update_parameters(self.network, step=iteration)

				if iteration % self.args.step_eval == 0 or iteration == self.args.iterations - 1:
					val_acc, val_loss = self.evaluate(iteration)

					if iteration > self.args.iterations * self.bash_args.start_swad:
						self.swad_valley.update_and_evaluate(self.swad_algorithm, val_acc, val_loss)
						if self.swad_valley.dead_valley:
							break
						self.swad_algorithm = swa_utils.AveragedModel(self.network)
			
			# if iteration % self.args.step_eval == 0 or iteration == self.args.iterations - 1:
			# 	print(
			# 		"Train set: Iteration: [{}/{}]\tClasification Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
			# 			iteration,
			# 			self.args.iterations,
			# 			n_class_corrected,
			# 			total_samples,
			# 			100.0 * n_class_corrected / total_samples,
			# 			cls_loss / total_samples,
			# 		)
			# 	)
			# 	print(
			# 		"Train set: Iteration: [{}/{}]\tDomain Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
			# 			iteration,
			# 			self.args.iterations,
			# 			n_domain_corrected,
			# 			total_samples,
			# 			100.0 * n_domain_corrected / total_samples,
			# 			cls_loss / total_samples,
			# 		)
			# 	)

			# 	print(
			# 		"Train set: Iteration: [{}/{}]\tPrototype Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
			# 			iteration,
			# 			self.args.iterations,
			# 			prototype_n_class_corrected,
			# 			prototype_total_samples,
			# 			100.0 * prototype_n_class_corrected / prototype_total_samples,
			# 			total_smooth_loss / prototype_total_samples,
			# 		)
			# 	)

			n_domain_corrected = 0
			n_class_corrected = 0
			prototype_n_class_corrected = 0
			total_samples = 0
			prototype_total_samples = 0
		

		self.network.eval()
		final_swad = self.swad_valley.get_final_model().module
		if not isinstance(final_swad, nn.Sequential):
			final_swad = final_swad.module
		self.save_model(final_swad, 'swad')
		self.save_model(self.network, 'last')
		test_acc, _ = self.evaluate_loader(final_swad, test_type='Test', model_type='swad')
		self.histogram()

	def histogram(self, data=None, labels=None):
	   
		self.hist = torch.zeros(len(self.train_loaders), self.num_embed).to(self.device)
		fault_map = 0
		prototype_feature = self.prototype_net.prototype
		for idx in range(len(self.train_loaders)):
			for iteration, (samples, labels, domains) in enumerate(self.train_loaders[idx]):
				samples = samples.to(self.device)
				labels = labels.to(self.device)
				domains = domains.to(self.device)
				features = self.network[0](samples) 
				cost_matrix = self.cosine_similarity(features, prototype_feature)
				sub_space_idx = cost_matrix.min(1)[1]
				fault_map = (1-(self.prototype_net.labels[sub_space_idx] == labels).int()).sum()
				for v in sub_space_idx:
					self.hist[idx, v]+=1
		
		print(self.hist.int())
		print(self.hist.sum(0).int())
		print('fault_map:' , fault_map)

	def save_model(self, network, name='best_model'):
		if self.bash_args.save_model_dir == '':
			return
		name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_en'.format(self.bash_args.model_name, name, 
				self.args.exp_name, 
				str(self.bash_args.buffer_size), 
				str(self.bash_args.prototype_per_class), 
				str(self.bash_args.ot_weight), 
				str(self.bash_args.smooth), 
				self.bash_args.exp_idx,
				self.bash_args.metric)
		
		torch.save({
			'encoder_state_dict': network[0].state_dict(), 
			'classifier_state_dict': network[1].state_dict(), 
			'prototype_state_dict': network[2].state_dict()}, 
			os.path.join(self.bash_args.save_model_dir, f'{name}_en.pth'))

	def load_model(self, ckpt):
		state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage)
		encoder_state = state_dict["encoder_state_dict"]
		classifier_state = state_dict["classifier_state_dict"]
		prototype_state = state_dict["prototype_state_dict"]
		self.encoder.load_state_dict(encoder_state)
		self.classifier.load_state_dict(classifier_state)
		self.prototype_net.load_state_dict(prototype_state)

	def evaluate_loader(self, model, test_type='Val', model_type='norm'):
		n_class_corrected = 0
		total_classification_loss = 0
		prototype_n_class_corrected = 0
		norm_prototype_n_class_corrected = 0
		ave_prototype_n_class_corrected = 0
		norm_ave_prototype_n_class_corrected = 0
		swad_prototype_n_class_corrected = 0
		swad_norm_prototype_n_class_corrected = 0
		swad_ave_prototype_n_class_corrected = 0
		swad_norm_ave_prototype_n_class_corrected = 0
		
		if test_type == 'Val':
			loader = self.val_loader
		else:
			loader = self.test_loader
		
	   
		with torch.no_grad():
			
			# import pdb; pdb.set_trace()
			
			# Using prototype
			prototype_logits = self.network[1](model[2].prototype)
			_, prototype_predicted_classes = torch.max(prototype_logits, 1)
			average_prototype = torch.zeros(self.args.n_classes, self.args.feature_dim).to(self.device)
			for i in range(self.args.n_classes):
				average_prototype[i]=model[2].prototype[prototype_predicted_classes==i].mean(0)

			prototype_logits = self.network[1](self.prototype_net.prototype)
			_, prototype_predicted_classes = torch.max(prototype_logits, 1)
			average_last_prototype = torch.zeros(self.args.n_classes, self.args.feature_dim).to(self.device)
			for i in range(self.args.n_classes):
				average_last_prototype[i]=self.prototype_net.prototype[prototype_predicted_classes==i].mean(0)


			for iteration, (samples, labels, domain_labels) in enumerate(loader):
				samples, labels = samples.to(self.device), labels.to(self.device)
				
				features = model[0](samples)
				
				# Using classifier for prediction
				predicted_classes = model[1](features)
				classification_loss = self.criterion(predicted_classes, labels)
				total_classification_loss += classification_loss.item()
				_, predicted_classes = torch.max(predicted_classes, 1)
				n_class_corrected += (predicted_classes == labels).sum().item()
				
				# Using protopye for prediction
				normed_domain_features = F.normalize(features, dim=1)
				normed_prototype = F.normalize(model[2].prototype, dim=1)
				
				# Option-1: Distance between normd prototype and feature
				tmp = torch.einsum('bd,dn->bn', normed_domain_features, rearrange(normed_prototype, 'n d -> d n'))
				cost_matrix = - tmp
				pred = prototype_predicted_classes[cost_matrix.min(1)[1]]
				norm_prototype_n_class_corrected += ((pred == labels).int()).sum().item()

				# Option-2: Distance between prototype and feature
				tmp = torch.einsum('bd,dn->bn', features, rearrange(model[2].prototype, 'n d -> d n'))
				cost_matrix = - tmp
				pred = prototype_predicted_classes[cost_matrix.min(1)[1]]
				prototype_n_class_corrected += ((pred == labels).int()).sum().item()
				

				# Using average protoypes of classes  for prediction
				predicted_classes = F.linear(features, average_prototype)
				_, predicted_classes = torch.max(predicted_classes, 1)
				ave_prototype_n_class_corrected += (predicted_classes == labels).sum().item()

				predicted_classes = F.linear(normed_domain_features, F.normalize(average_prototype,  dim=1))
				_, predicted_classes = torch.max(predicted_classes, 1)
				norm_ave_prototype_n_class_corrected += (predicted_classes == labels).sum().item()
				

				# Using swad-encoder with last prototype istead of swad-prototype
				if model_type == 'swad':
					normed_domain_features = F.normalize(features, dim=1)
					normed_prototype = F.normalize(self.prototype_net.prototype, dim=1)
					tmp = torch.einsum('bd,dn->bn', normed_domain_features, rearrange(normed_prototype, 'n d -> d n'))
					cost_matrix = - tmp
					pred = prototype_predicted_classes[cost_matrix.min(1)[1]]
					swad_norm_prototype_n_class_corrected += ((pred == labels).int()).sum().item()
					
					tmp = torch.einsum('bd,dn->bn', features, rearrange(self.prototype_net.prototype, 'n d -> d n'))
					cost_matrix = - tmp
					pred = prototype_predicted_classes[cost_matrix.min(1)[1]]
					swad_prototype_n_class_corrected += ((pred == labels).int()).sum().item()

					predicted_classes = F.linear(features, average_last_prototype)
					_, predicted_classes = torch.max(predicted_classes, 1)
					swad_ave_prototype_n_class_corrected += (predicted_classes == labels).sum().item()

					predicted_classes = F.linear(normed_domain_features, F.normalize(average_last_prototype,  dim=1))
					_, predicted_classes = torch.max(predicted_classes, 1)
					swad_norm_ave_prototype_n_class_corrected += (predicted_classes == labels).sum().item()
					
		print("-----------------------------------")
		print_out = "{} set: Accuracy: {}/{} {:.2f}%, {:.2f}% , {:.2f}% , {:.2f}% , {:.2f}%, {}".format(
			test_type, 
			n_class_corrected,len(loader.dataset),
			100.0 * n_class_corrected / len(loader.dataset),
			100.0 * prototype_n_class_corrected / len(loader.dataset), 
			100.0 * norm_prototype_n_class_corrected / len(loader.dataset), 
			100.0 * ave_prototype_n_class_corrected / len(loader.dataset), 
			100.0 * norm_ave_prototype_n_class_corrected / len(loader.dataset), 
			total_classification_loss / len(loader.dataset),)
		print(print_out)
		print("-----------------------------------")

		
		# import pdb; pdb.set_trace()

		if model_type == 'swad':
			print("-----------------------------------")
			print_out_swad = "{} set: SWAD-encoder + Last-protype Accuracy: {}/{} {:.2f}%, {:.2f}% , {:.2f}% , {:.2f}% , {:.2f}%, {}".format(
				test_type, 
				n_class_corrected,len(loader.dataset),
				100.0 * n_class_corrected / len(loader.dataset),
				100.0 * swad_prototype_n_class_corrected / len(loader.dataset), 
				100.0 * swad_norm_prototype_n_class_corrected / len(loader.dataset), 
				100.0 * swad_ave_prototype_n_class_corrected / len(loader.dataset), 
				100.0 * swad_norm_ave_prototype_n_class_corrected / len(loader.dataset), 
				total_classification_loss / len(loader.dataset),)
			print(print_out_swad)
			print("-----------------------------------")
			
			# name = '{}_{}_ot_{}_smooth_{}_pcl-type_{}_{}'.format(self.args.dataset, 
			# 	str(self.bash_args.prototype_per_class), 
			# 	str(self.bash_args.ot_weight), 
			# 	str(self.bash_args.smooth), 
			# 	str(self.bash_args.pcl_weight), 
			# 	str(self.bash_args.pcl_norm))

			# f = open("algorithms/BAIR/results/no-pretrained_{}.txt".format(name), "a")
			# f.write("{}, seed {}".format(self.args.exp_name, self.bash_args.exp_idx))

			# f.write('\n')
			# f.write(print_out)
			# f.write('\n')
			# f.write(print_out_swad)
			# f.write('\n')
			# f.close()

		return n_class_corrected / len(loader.dataset), total_classification_loss / len(loader.dataset)
		

	def evaluate(self, n_iter):
		self.network.eval()
		
		val_acc, val_loss = self.evaluate_loader(self.network, test_type='Val')
		test_acc, _ = self.evaluate_loader(self.network, test_type='Test')
	 
		if self.val_acc_max < val_acc:
			self.val_acc_max = val_acc
			self.corresponding_test = test_acc
			self.save_model(self.network, 'corr')
			
		if self.test_acc_max < test_acc:
			self.test_acc_max = test_acc
			self.save_model(self.network, 'best')

		print( "Best val: {:.2f}%, Corres: {:.2f}%, Best Test: {:.2f}%".format(
			100.0 * self.val_acc_max, 100.0 * self.corresponding_test, 100.0 * self.test_acc_max))
		
		self.network.train()
		return val_acc, val_loss

	def test(self):
		self.network.eval()
		val_acc, _ = self.evaluate_loader(self.network, test_type='Val')
		test_acc, _ = self.evaluate_loader(self.network, test_type='Test')
		print("val: {}, Test: {}".format(val_acc, test_acc))

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

					features = self.encoder(samples)
					
					feature_train += features.tolist()
					Y_train += labels.tolist()
					Y_domain_train += domain_labels.tolist()
			print("Train dumped")
			
			for iteration, (samples, labels, domain_labels) in enumerate(self.test_loader):
				samples, labels = samples.to(self.device), labels.to(self.device)
				features = self.encoder(samples)
				feature_test += features.tolist()
				Y_test += labels.tolist()
				Y_domain_test += domain_labels.tolist()
			print("Test dumped")

		if not os.path.exists(plot_dir):
			os.mkdir(plot_dir)
		if name_folder is None:
			name_folder = "{}_seed_{}".format(self.args.exp_name, self.bash_args.exp_idx)
		fol_name = os.path.join(plot_dir, name_folder)
		# plot_dir += self.args.exp_name + "_seed_"+ self.bash_args.exp_idx + "_"
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
