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
import torchvision.transforms as transforms

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, GradReverse, EntropyWeight

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

def to_minibatch_d(x, y, d):
	minibatches = list(zip(x, y, d))
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

class ERM(Algorithm):
	"""
	Empirical Risk Minimization (ERM)
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(ERM, self).__init__(input_shape, num_classes, num_domains,
								  hparams)
		self.featurizer = networks.Featurizer(input_shape, self.hparams)
		self.classifier = networks.Classifier(
			self.featurizer.n_outputs,
			num_classes)

		self.network = nn.Sequential(self.featurizer, self.classifier)
		self.optimizer = torch.optim.Adam(
			self.network.parameters(),
			lr=self.hparams["lr"],
			weight_decay=self.hparams['weight_decay']
		)

	def update(self, x, y, **kwargs):
		minibatches = to_minibatch(x, y)
		all_x = torch.cat([x for x, y in minibatches])
		all_y = torch.cat([y for x, y in minibatches])
		loss = F.cross_entropy(self.predict(all_x), all_y)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return {'loss': loss.item()}

	def predict(self, x):
		return self.network(x)


class Classifier(nn.Module):
	def __init__(self, feature_dim, n_classes):
		super(Classifier, self).__init__()
		self.dropout = nn.Dropout(0.5)
		self.linear = nn.Linear(int(feature_dim), n_classes, bias=False)

	def forward(self, x):
		x = self.dropout(x)
		y = self.linear(x)
		return y

class Prototype(nn.Module):
	def __init__(self, num_embed, n_classes, n_domains, prototype_per_class, feature_dim=2048):
		super(Prototype, self).__init__()
		self.n_classes = n_classes
		self.feature_dim = feature_dim
		self.n_domains = n_domains
		self.num_embed = num_embed
		self.prototype = nn.Parameter(torch.ones(num_embed, feature_dim))
		self.prototype.data.uniform_(-1.0 /  num_embed, 1.0 /  num_embed)
		self.register_buffer('prototype_weight', torch.ones(num_embed) / num_embed)
		self.register_buffer('labels', torch.ones(num_embed))
		self.criterion = nn.CrossEntropyLoss()
		self.warmup_finised = False

	def forward(self, features, labels, classifer_weight, lambda_ot=0.1):
		subspace_projection_loss = 0
		sample_weight = torch.ones(features.shape[0]).to(features.device) / features.shape[0]
		cost_matrix = self.cost_matrix(features, self.prototype)
		ot_cost = ot.emd2(sample_weight, self.prototype_weight, cost_matrix, numItermax=500000)
		subspace_projection_loss += ot_cost
		nearest_subspace = cost_matrix.min(1)[1]

		quantized_feature = self.prototype[nearest_subspace]
		quantized_logits = F.linear(quantized_feature, classifer_weight)
		quantized_loss = self.criterion(quantized_logits, labels)
		predicted_prototype = F.linear(features, self.prototype)
		prototype_loss = self.criterion(predicted_prototype, nearest_subspace)
		if self.warmup_finised:
			subspace_projection_loss += lambda_ot * (prototype_loss)/self.n_domains
			subspace_projection_loss += lambda_ot * (quantized_loss)/self.n_domains
		
		return subspace_projection_loss, nearest_subspace, nn.Softmax(dim=1)(predicted_prototype)

	def update_label(self, classifer):
		# Test whether prototype can be used to make predition
		prototype_logits = classifer(self.prototype)
		_, predicted_classes = torch.max(prototype_logits, 1)
		self.labels = predicted_classes
		return prototype_logits

	def average_prototype(self):
	# Using prototype
		average_prototype = torch.zeros(self.n_classes, self.feature_dim).to(self.prototype.device)
		for i in range(self.n_classes):
			average_prototype[i]=self.prototype[self.labels==i].mean(0)
		return average_prototype
	
	def cost_matrix(self, feature_A, feature_B, metric='cosine'):
		feature_A = F.normalize(feature_A, dim=1)
		feature_B = F.normalize(feature_B, dim=1)
		pair_wises = torch.einsum('bd,dn->bn', feature_A, rearrange(feature_B, 'n d -> d n'))
		similarity = 1 - pair_wises
		return similarity


class SRA(Algorithm):

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(SRA, self).__init__(input_shape, num_classes, num_domains, hparams)
		self.hparams = hparams
		self.n_classes = num_classes
		self.n_domain_classes = num_domains
		self.featurizer = networks.Featurizer(input_shape, self.hparams)
		self.feature_dim = self.featurizer.n_outputs

		
		self.classifier = Classifier(feature_dim=self.feature_dim, n_classes=self.n_classes)

		# Define Prototypes
		self.num_embed = self.n_classes * self.hparams['prototype_per_class']
		self.prototype_net = Prototype(self.num_embed, self.n_classes, self.n_domain_classes, self.hparams['prototype_per_class'], self.feature_dim)
		
		# Define Disciminator for Prototype_DANN
		self.alpha = torch.tensor(hparams['disc_weight'], requires_grad=False)
		self.discriminator = networks.MLP(self.feature_dim, self.n_domain_classes, hparams)
		self.subspace_embeddings = nn.Embedding(self.num_embed, self.feature_dim)

		self.network = nn.Sequential(self.featurizer, self.classifier, self.prototype_net)  
		
		self.optimizer = torch.optim.Adam(
			list(self.prototype_net.parameters())+
			list(self.featurizer.parameters())+
			list(self.classifier.parameters()),
			lr=self.hparams["lr"], 
			weight_decay=self.hparams["weight_decay"])

		self.disc_optimizer = torch.optim.Adam(
			list(self.subspace_embeddings.parameters())+
			list(self.discriminator.parameters()), 
			lr=self.hparams["lr"], 
			weight_decay=self.hparams["weight_decay"],
			betas=(0.5, 0.9))

		self.criterion = nn.CrossEntropyLoss()
		self.update_count = 0
		self.batch_size = self.hparams['batch_size']
		self.n_domain_corrected = 0

	def reset_optimizer(self):
		self.optimizer = torch.optim.Adam(
			list(self.prototype_net.parameters())+
			list(self.featurizer.parameters())+
			list(self.classifier.parameters()),
			lr=self.hparams["lr"], 
			weight_decay=self.hparams["weight_decay"])

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

		features = self.network[0](tr_samples)
		if self.update_count < self.hparams['warm_up']:
			features = features.detach()
		
		if self.update_count == self.hparams['warm_up']:
			self.reset_optimizer()
			self.prototype_net.warmup_finised = True

		logits = self.network[1](features)
		cls_loss = self.criterion(logits, tr_labels)
		total_loss += cls_loss

		total_domain_loss = 0.0
		for d_index in range(self.n_domain_classes): 
			
			# Sub-space projetion via Wasserstein with different metric
			domain_features = features[self.batch_size*d_index:self.batch_size*(d_index+1),:]
			subspace_projection_loss, nearest_subspace, assgined_confident_score = self.prototype_net(
				domain_features, 
				tr_labels[self.hparams['batch_size']*d_index:self.hparams['batch_size']*(d_index+1)],
				self.network[1].linear.weight)

			total_loss += self.hparams['ot_weight'] * subspace_projection_loss

			# sub-space balanced alignment via Prototype-DANN
			disc_input = GradReverse.apply(features[self.batch_size*d_index:self.batch_size*(d_index+1)], self.alpha) + self.subspace_embeddings(nearest_subspace)
			domain_logit =  self.discriminator(disc_input)
			domain_loss = F.cross_entropy(domain_logit, tr_domain_labels[self.batch_size*d_index:self.batch_size*(d_index+1)], reduction='none')
			
			# larger weight on features which are confident with its protytype
			weight = EntropyWeight(assgined_confident_score, self.hparams['disc_weight'])
			domain_loss = torch.clamp((weight * domain_loss).sum(), max=10.0)
			total_domain_loss += domain_loss / self.n_domain_classes

			_, predicted_domain = torch.max(domain_logit, 1)

		total_loss += total_domain_loss
		self.optimizer.zero_grad()
		self.disc_optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		self.disc_optimizer.step()      
		return {"loss": total_loss.item(), "domain_loss": total_domain_loss.item()}


	def predict(self, x):
		z = self.network[0](x)
		y = self.network[1](z)
		return y

	def tuning(self, x, y, **kwargs):
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
		features = self.network[0](tr_samples)
		features = features.detach()
		logits = self.network[1](features)
		cls_loss = self.criterion(logits, tr_labels)
		total_loss += cls_loss
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()
		return {"loss": total_loss.item()}


class AbstractDANN(Algorithm):
	"""Domain-Adversarial Neural Networks (abstract class)"""

	def __init__(self, input_shape, num_classes, num_domains,
				 hparams, conditional, class_balance):

		super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
								  hparams)

		self.register_buffer('update_count', torch.tensor([0]))
		self.conditional = conditional
		self.class_balance = class_balance

		# Algorithms
		self.featurizer = networks.Featurizer(input_shape, self.hparams)
		self.classifier = networks.Classifier(
			self.featurizer.n_outputs,
			num_classes)

		self.network = nn.Sequential(self.featurizer, self.classifier)  
		self.discriminator = networks.MLP(self.featurizer.n_outputs,
			num_domains, self.hparams)
		self.class_embeddings = nn.Embedding(num_classes,
			self.featurizer.n_outputs)

		# Optimizers
		self.disc_opt = torch.optim.Adam(
			(list(self.discriminator.parameters()) +
				list(self.class_embeddings.parameters())),
			lr=self.hparams["lr_d"],
			weight_decay=self.hparams['weight_decay_d'],
			betas=(self.hparams['beta1'], 0.9))

		self.gen_opt = torch.optim.Adam(
			(list(self.featurizer.parameters()) +
				list(self.classifier.parameters())),
			lr=self.hparams["lr_g"],
			weight_decay=self.hparams['weight_decay_g'],
			betas=(self.hparams['beta1'], 0.9))

	def update(self, x, y, **kwargs):
		minibatches = to_minibatch(x, y)
		device = "cuda" if minibatches[0][0].is_cuda else "cpu"
		self.update_count += 1
		all_x = torch.cat([x for x, y in minibatches])
		all_y = torch.cat([y for x, y in minibatches])

		all_z = self.featurizer(all_x)
		if self.conditional:
			disc_input = all_z + self.class_embeddings(all_y)
		else:
			disc_input = all_z
		disc_out = self.discriminator(disc_input)
		disc_labels = torch.cat([
			torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
			for i, (x, y) in enumerate(minibatches)
		])

		if self.class_balance:
			y_counts = F.one_hot(all_y).sum(dim=0)
			weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
			disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
			disc_loss = (weights * disc_loss).sum()
		else:
			disc_loss = F.cross_entropy(disc_out, disc_labels)

		input_grad = autograd.grad(
			F.cross_entropy(disc_out, disc_labels, reduction='sum'),
			[disc_input], create_graph=True)[0]
		grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
		disc_loss += self.hparams['grad_penalty'] * grad_penalty

		d_steps_per_g = self.hparams['d_steps_per_g_step']
		if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

			self.disc_opt.zero_grad()
			disc_loss.backward()
			self.disc_opt.step()
			return {'disc_loss': disc_loss.item()}
		else:
			all_preds = self.classifier(all_z)
			classifier_loss = F.cross_entropy(all_preds, all_y)
			gen_loss = (classifier_loss +
						(self.hparams['lambda'] * -disc_loss))
			self.disc_opt.zero_grad()
			self.gen_opt.zero_grad()
			gen_loss.backward()
			self.gen_opt.step()
			return {'gen_loss': gen_loss.item()}

	def predict(self, x):
		return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
	"""Unconditional DANN"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(DANN, self).__init__(input_shape, num_classes, num_domains,
			hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
	"""Conditional DANN"""
	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(CDANN, self).__init__(input_shape, num_classes, num_domains,
			hparams, conditional=True, class_balance=True)

class IRM(ERM):
	"""Invariant Risk Minimization"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
		self.register_buffer("update_count", torch.tensor([0]))

	@staticmethod
	def _irm_penalty(logits, y):
		scale = torch.tensor(1.0).cuda().requires_grad_()
		loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
		loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
		grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
		grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
		result = torch.sum(grad_1 * grad_2)
		return result

	def update(self, x, y, **kwargs):
		minibatches = to_minibatch(x, y)
		penalty_weight = (
			self.hparams["irm_lambda"]
			if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
			else 1.0
		)
		nll = 0.0
		penalty = 0.0

		all_x = torch.cat([x for x, y in minibatches])
		all_logits = self.network(all_x)
		all_logits_idx = 0
		for i, (x, y) in enumerate(minibatches):
			logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
			all_logits_idx += x.shape[0]
			nll += F.cross_entropy(logits, y)
			penalty += self._irm_penalty(logits, y)
		nll /= len(minibatches)
		penalty /= len(minibatches)
		loss = nll + (penalty_weight * penalty)

		if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
			# Reset Adam, because it doesn't like the sharp jump in gradient
			# magnitudes that happens at this step.
			self.optimizer = get_optimizer(
				self.hparams["optimizer"],
				self.network.parameters(),
				lr=self.hparams["lr"],
				weight_decay=self.hparams["weight_decay"],
			)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.update_count += 1
		return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
	"""V-REx algorithm from http://arxiv.org/abs/2003.00688"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
		self.register_buffer("update_count", torch.tensor([0]))

	def update(self, x, y, **kwargs):
		minibatches = to_minibatch(x, y)
		if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
			penalty_weight = self.hparams["vrex_lambda"]
		else:
			penalty_weight = 1.0

		nll = 0.0

		all_x = torch.cat([x for x, y in minibatches])
		all_logits = self.network(all_x)
		all_logits_idx = 0
		losses = torch.zeros(len(minibatches))
		for i, (x, y) in enumerate(minibatches):
			logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
			all_logits_idx += x.shape[0]
			nll = F.cross_entropy(logits, y)
			losses[i] = nll

		mean = losses.mean()
		penalty = ((losses - mean) ** 2).mean()
		loss = mean + penalty_weight * penalty

		if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
			# Reset Adam (like IRM), because it doesn't like the sharp jump in
			# gradient magnitudes that happens at this step.
			self.optimizer = get_optimizer(
				self.hparams["optimizer"],
				self.network.parameters(),
				lr=self.hparams["lr"],
				weight_decay=self.hparams["weight_decay"],
			)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.update_count += 1
		return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class AbstractMMD(ERM):
	"""
	Perform ERM while matching the pair-wise domain feature distributions
	using MMD (abstract class)
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
		super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
		if gaussian:
			self.kernel_type = "gaussian"
		else:
			self.kernel_type = "mean_cov"

	def my_cdist(self, x1, x2):
		x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
		x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
		res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
			x1_norm
		)
		return res.clamp_min_(1e-30)

	def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
		D = self.my_cdist(x, y)
		K = torch.zeros_like(D)

		for g in gamma:
			K.add_(torch.exp(D.mul(-g)))

		return K

	def mmd(self, x, y):
		if self.kernel_type == "gaussian":
			Kxx = self.gaussian_kernel(x, x).mean()
			Kyy = self.gaussian_kernel(y, y).mean()
			Kxy = self.gaussian_kernel(x, y).mean()
			return Kxx + Kyy - 2 * Kxy
		else:
			mean_x = x.mean(0, keepdim=True)
			mean_y = y.mean(0, keepdim=True)
			cent_x = x - mean_x
			cent_y = y - mean_y
			cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
			cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

			mean_diff = (mean_x - mean_y).pow(2).mean()
			cova_diff = (cova_x - cova_y).pow(2).mean()

			return mean_diff + cova_diff

	def update(self, x, y, **kwargs):
		minibatches = to_minibatch(x, y)
		objective = 0
		penalty = 0
		nmb = len(minibatches)

		features = [self.featurizer(xi) for xi, _ in minibatches]
		classifs = [self.classifier(fi) for fi in features]
		targets = [yi for _, yi in minibatches]

		for i in range(nmb):
			objective += F.cross_entropy(classifs[i], targets[i])
			for j in range(i + 1, nmb):
				penalty += self.mmd(features[i], features[j])

		objective /= nmb
		if nmb > 1:
			penalty /= nmb * (nmb - 1) / 2

		self.optimizer.zero_grad()
		(objective + (self.hparams["mmd_gamma"] * penalty)).backward()
		self.optimizer.step()

		if torch.is_tensor(penalty):
			penalty = penalty.item()

		return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
	"""
	MMD using Gaussian kernel
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
	"""
	MMD using mean and covariance difference
	"""

	def __init__(self, input_shape, num_classes, num_domains, hparams):
		super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)

