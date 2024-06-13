# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch import einsum
from einops import rearrange
import ot
#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
	resnet18_mixstyle_L234_p0d5_a0d1,
	resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
	resnet18_mixstyle2_L234_p0d5_a0d1,
	resnet50_mixstyle2_L234_p0d5_a0d1,
)


def to_minibatch(x, y):
	minibatches = list(zip(x, y))
	return minibatches


class Algorithm(torch.nn.Module):
	"""
	A subclass of Algorithm implements a domain generalization algorithm.
	Subclasses should implement the following:
	- update()
	- predict()
	"""

	transforms = {}

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(Algorithm, self).__init__()
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.num_domains = num_domains
		self.hparams = hparams

	def update(self, x, y, **kwargs):
		"""
		Perform one update step, given a list of (x, y) tuples for all
		environments.
		"""
		raise NotImplementedError

	def predict(self, x):
		raise NotImplementedError

	def forward(self, x):
		return self.predict(x)

	def new_optimizer(self, parameters):
		optimizer = get_optimizer(
			self.hparams["optimizer"],
			parameters,
			lr=self.hparams["lr"],
			weight_decay=self.hparams["weight_decay"],
		)
		return optimizer

	def clone(self):
		clone = copy.deepcopy(self)
		clone.optimizer = self.new_optimizer(clone.network.parameters())
		clone.optimizer.load_state_dict(self.optimizer.state_dict())

		return clone


class Classifier(nn.Module):
	def __init__(self, feature_dim, n_classes):
		super(Classifier, self).__init__()
		self.classifier = nn.Linear(int(feature_dim), n_classes, bias=False)

	def forward(self, x):
		y = self.classifier(x)
		return y


def grl_hook(coeff):
	def fun1(grad):
		return -coeff*grad.clone()
	return fun1


def Entropy(input_):
	bs = input_.size(0)
	epsilon = 1e-5
	entropy = -input_ * torch.log(input_ + epsilon)
	entropy = torch.sum(entropy, dim=1)
	return entropy 


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


class Prototype(nn.Module):
	def __init__(self, num_embed, n_classes, prototype_per_class, feature_dim=2048):
		super(Prototype, self).__init__()
		self.n_classes = n_classes
		self.num_embed = num_embed
		self.weight = nn.Parameter(torch.ones(num_embed)/num_embed)
		self.prototype = nn.Parameter(torch.ones(num_embed, feature_dim))
		self.prototype.data.uniform_(-1.0 /  num_embed, 1.0 /  num_embed)
		self.register_buffer('labels', torch.tensor(range(n_classes)).unsqueeze(-1).repeat(1, prototype_per_class).reshape(-1))

	def forward(self, batch_classes):
		mask = (self.labels.unsqueeze(1)==batch_classes.unsqueeze(0)).sum(1) > 0
		selected_prototypes = self.prototype[torch.where(mask > 0)]
		selected_prototype_weight = self.weight[torch.where(mask > 0)]
		selected_prototype_weight = nn.Softmax(dim=0)(selected_prototype_weight)
		return selected_prototypes, selected_prototype_weight
	
	def update_label(self, predicted_classes):
		self.labels = predicted_classes


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
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
		feature = feature * ~label_matrix  # get negative matrix
		feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)
		
		logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
		label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
		loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
		
		return loss

class ERM(Algorithm):
	"""
	Empirical Risk Minimization (ERM)
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
		self.featurizer = networks.Featurizer(input_shape, self.hparams)
		
		self.prototype_per_class = hparams['prototype_per_class']
		self.n_classes = num_classes
		self.n_domain_classes = num_domains
		self.feature_dim = self.featurizer.n_outputs
		self.batch_size = hparams['batch_size']

		self.warm_up = hparams['warm_up']
		self.disc_weight = hparams['disc_weight']
		self.maxinfo_weight = hparams['maxinfo_weight']
		self.ot_weight = hparams['ot_weight']
		self.clip_disc = hparams['clip_disc']
		

		self.classifier = Classifier(feature_dim=self.featurizer.n_outputs, 
									 n_classes=self.n_classes)
		self.invariant_classifier = Classifier(feature_dim=self.featurizer.n_outputs+1, 
									 n_classes=self.n_classes)

		
		# Define Prototypes
		self.num_embed = self.n_classes * self.prototype_per_class
		self.prototype_net = Prototype(self.num_embed, self.n_classes, self.prototype_per_class, self.feature_dim)
		self.scale = torch.tensor(-self.ot_weight, requires_grad=False)

		
		# Define Disciminator for Prototype_DANN
		self.alpha = torch.tensor(self.disc_weight, requires_grad=False)
		self.discriminator = networks.MLP(self.feature_dim, self.n_domain_classes, hparams)
		self.subspace_embeddings = nn.Embedding(self.num_embed, self.feature_dim)


		self.network = nn.Sequential(self.featurizer, self.classifier, self.prototype_net)  
		
		self.optimizer = torch.optim.Adam(
			list(self.prototype_net.parameters())+
			list(self.featurizer.parameters())+
			list(self.invariant_classifier.parameters())+
			list(self.classifier.parameters()),
			lr=self.hparams["lr"], 
			weight_decay=self.hparams["weight_decay"])


		self.disc_optimizer = torch.optim.Adam(
			list(self.subspace_embeddings.parameters())+
			list(self.discriminator.parameters()), 
			lr=self.hparams["lr"], 
			weight_decay=self.hparams["weight_decay"],
			betas=(0.5, 0.9))


		self.maxinfo_loss = MaxInfo(num_classes)
		self.criterion = nn.CrossEntropyLoss()
		self.update_count = 0


	# METRIC for WS ----------------------------------------------
	def cosine_similarity(self, domain_features, prototype_feature):
		normed_domain_features = F.normalize(domain_features, dim=1)
		normed_prototype = F.normalize(prototype_feature, dim=1)
		pair_wises = torch.einsum('bd,dn->bn', normed_domain_features, rearrange(normed_prototype, 'n d -> d n'))
		similarity = 1 - pair_wises
		return similarity
	# METRIC for WS ----------------------------------------------
	

	def update(self, x, y, **kwargs):

		minibatches = to_minibatch(x, y)
		device = "cuda" if minibatches[0][0].is_cuda else "cpu"
		self.update_count += 1
		tr_samples = torch.cat([x for x, y in minibatches])
		tr_labels = torch.cat([y for x, y in minibatches])
		
		tr_domain_labels = torch.cat([
			torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
			for i, (x, y) in enumerate(minibatches)
		])


		total_loss = 0
		# Classification Loss
		features = self.network[0](tr_samples)
		logits = self.network[1](features)
		cls_loss = self.criterion(logits, tr_labels)
		total_loss += cls_loss
		

		# Enforcing Invariant Prediction Loss (IRM)
		self.domain_indx = [torch.full((self.batch_size, 1), indx).to(device) for indx in range(self.n_domain_classes)]
		embeddings = torch.cat([curr_dom_embed for curr_dom_embed in self.domain_indx]).to(device)
		env_logits = self.invariant_classifier(torch.cat([features, embeddings], 1))
		env_loss = self.criterion(env_logits, tr_labels)
		inv_loss = (cls_loss - env_loss) ** 2 
		total_loss += env_loss + inv_loss


		# Test whether prototype can be used to make predition
		prototype_logits = self.network[1](self.prototype_net.prototype)
		_, prototype_predicted_classes = torch.max(prototype_logits, 1)

		# Assign labels to prototypes
		if self.update_count > self.warm_up:
			self.prototype_net.update_label(prototype_predicted_classes)
		
		predicted_prototype = F.linear(features, self.prototype_net.prototype)
		softmax_prototype = nn.Softmax(dim=1)(predicted_prototype)
			
		# Feature regularization via contrastive learning -> maximun I(g(X),X) for each source domains	
		for d_index in range(self.n_domain_classes): 
			normed_domain_features = F.normalize(features[self.batch_size*d_index:self.batch_size*(d_index+1),:], dim=1)
			domain_class_labels = tr_labels[self.batch_size*d_index:self.batch_size*(d_index+1)]
			loss_info = self.maxinfo_loss(normed_domain_features, domain_class_labels, F.normalize(self.classifier.classifier.weight, dim=1))
			total_loss += self.maxinfo_weight * loss_info / self.n_domain_classes
				

		# Sub-space projetion via Wasserstein with different metric
		for d_index in range(self.n_domain_classes): 
			
			domain_features = features[self.batch_size*d_index:self.batch_size*(d_index+1),:]
			batch_d_labels = tr_labels[self.batch_size*d_index:self.batch_size*(d_index+1)]
			domain_unique_class = batch_d_labels.unique()
			
			prototype_feature, prototype_weight = self.prototype_net(domain_unique_class)

			sample_weight = torch.ones(domain_features.shape[0]).to(device) / domain_features.shape[0]
			cost_matrix = self.cosine_similarity(GradReverse.apply(domain_features, self.scale), prototype_feature)
			ot_cost = ot.emd2(sample_weight, prototype_weight.detach(), cost_matrix, numItermax=500000, return_matrix=True)
			total_loss += self.ot_weight * ot_cost[0]
			

			# sub-space balanced alignment via Prototype-DANN
			sub_space_idx = cost_matrix.min(1)[1]
			disc_input = GradReverse.apply(features[self.batch_size*d_index:self.batch_size*(d_index+1)], self.alpha) + self.subspace_embeddings(sub_space_idx)
			if self.update_count > self.warm_up:
				domain_logit =  self.discriminator(disc_input)
			else:
				domain_logit =  self.discriminator(disc_input.detach())
			domain_loss = F.cross_entropy(domain_logit, tr_domain_labels[self.batch_size*d_index:self.batch_size*(d_index+1)], reduction='none')

			if domain_loss.mean().item() > self.clip_disc and self.update_count > self.warm_up:
				domain_logit =  self.discriminator(disc_input.detach())
				domain_loss = F.cross_entropy(domain_logit, tr_domain_labels[self.batch_size*d_index:self.batch_size*(d_index+1)], reduction='none')

			softmax_output = softmax_prototype[self.batch_size*d_index:self.batch_size*(d_index+1)]
			entropy = Entropy(softmax_output)
			entropy.register_hook(grl_hook(self.disc_weight))
			entropy = 1.0+torch.exp(-entropy)
			weight = entropy / torch.sum(entropy).detach().item()
			domain_loss = (weight * domain_loss).sum()
			
			total_loss += domain_loss / self.n_domain_classes
			_, predicted_domain = torch.max(domain_logit, 1)


		self.optimizer.zero_grad()
		self.disc_optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		self.disc_optimizer.step()      
		return {"loss": total_loss.item()}

	def predict(self, x, average_prototype=None):
		z = self.network[0](x)
		if average_prototype is None:
			y = self.network[1](z)
		else:
			y = F.linear(F.normalize(z, dim=1), F.normalize(average_prototype,  dim=1))
		return y


