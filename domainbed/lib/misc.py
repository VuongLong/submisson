# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import os
import shutil
import errno
from datetime import datetime
from collections import Counter
import copy

import numpy as np
import torch
import torch.nn as nn
import domainbed.lib.augmentations as augmentations


def grl_hook(coeff):
	def fun1(grad):
		return -coeff*grad.clone()
	return fun1

def EntropyWeight(input_, coeff=1.0):
	bs = input_.size(0)
	epsilon = 1e-5
	entropy = -input_ * torch.log(input_ + epsilon)
	entropy = torch.sum(entropy, dim=1)
	entropy.register_hook(grl_hook(coeff))
	entropy = 1.0+torch.exp(-entropy)
	weight = entropy / torch.sum(entropy).detach().item()
	return weight 


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

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def to_row(row, colwidth=10, latex=False):
    """Convert value list to row string"""
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.6f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + " " + end_


def random_pairs_of_minibatches(minibatches):
    # n_tr_envs = len(minibatches)
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        # j = cyclic(i + 1)
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


###########################################################
# Custom utils
###########################################################


def index_conditional_iterate(skip_condition, iterable, index):
    for i, x in enumerate(iterable):
        if skip_condition(i):
            continue

        if index:
            yield i, x
        else:
            yield x


class SplitIterator:
    def __init__(self, test_envs):
        self.test_envs = test_envs

    def train(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx in self.test_envs, iterable, index)

    def test(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx not in self.test_envs, iterable, index)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} (val={:.3f}, count={})".format(self.avg, self.val, self.count)


class AverageMeters:
    def __init__(self, *keys):
        self.keys = keys
        for k in keys:
            setattr(self, k, AverageMeter())

    def resets(self):
        for k in self.keys:
            getattr(self, k).reset()

    def updates(self, dic, n=1):
        for k, v in dic.items():
            getattr(self, k).update(v, n)

    def __repr__(self):
        return "  ".join(["{}: {}".format(k, str(getattr(self, k))) for k in self.keys])

    def get_averages(self):
        dic = {k: getattr(self, k).avg for k in self.keys}
        return dic


def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def rm(path):
    """remove dir recursively"""
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def cp(src, dst):
    shutil.copy2(src, dst)


def get_lr(optimizer):
    """Assume that the optimizer has single lr"""
    lr = optimizer.param_groups[0]["lr"]

    return lr


@torch.no_grad()
def hash_bn(module):
    summary = []
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            w = m.weight.detach().mean().item()
            b = m.bias.detach().mean().item()
            rm = m.running_mean.detach().mean().item()
            rv = m.running_var.detach().mean().item()
            summary.append((w, b, rm, rv))
    w, b, rm, rv = [np.mean(col) for col in zip(*summary)]

    return w, b, rm, rv


def merge_dictlist(dictlist):
    """Merge list of dicts into dict of lists, by grouping same key."""
    ret = {k: [] for k in dictlist[0].keys()}
    for dic in dictlist:
        for data_key, v in dic.items():
            ret[data_key].append(v)
    return ret

augmentations.IMAGE_SIZE = 224
def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    mixture_width = 3
    mixture_depth = -1
    aug_severity = 1
    ws = np.float32(
        np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed

def Augmix(x, preprocess, no_jsd):
    if no_jsd:
      return aug(x, preprocess)
    else:
      return preprocess(x), aug(x, preprocess), aug(x, preprocess)