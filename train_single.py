import argparse
import json
import os
import random

import numpy as np
import torch
from domainbed.trainer import Trainer
from domainbed import hparams_registry


def fix_random_seed(seed_value):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_value)
		torch.cuda.manual_seed(seed_value)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


def get_dataset_configs():
	parser = argparse.ArgumentParser()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--algorithm", type=str, default="SRA")
	parser.add_argument("--data_dir", help="data path")
	parser.add_argument("--dataset", help="dataset")
	parser.add_argument("--target", help="target domain")
	parser.add_argument("--seed", type=int, default=1, help="Index of experiment")

	# swad: average moving model
	parser.add_argument("--start_swad", type=float, default=0.0, help="For large datasets, perform validation only after the training has converged to save time.")
	parser.add_argument("--checkpoint_freq", type=int, default=300, help="Save checkpoints and update SWAD at every specified number of iterations.")
	parser.add_argument("--save_model_dir", type=str, default='', help="The model will not be saved if no directory is specified.")
	parser.add_argument("--ckpt", type=str, default='', help="Path to the checkpoint being evaluated.")
	parser.add_argument("--plotdir", type=str, default='', help="The features will be saved if directory is specified.")

	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# setup hparams

	hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

	with open('domainbed/configs/{}_{}.json'.format(args.dataset, args.target), "r") as inp:
		dataset_configs = argparse.Namespace(**json.load(inp))
	dataset_configs.src_data_path = str(args.data_dir) +'/' + dataset_configs.dataset + '/'

	print(args)
	print(dataset_configs)
	print(hparams)
	
	seeds = [0,1,2]
	if args.ckpt == '':
		for seed in seeds:
			args.seed = seed
			# args.save_model_dir = 'ICLR_model/'

			# if seed == 0:
			# 	args.save_model_dir = 'ICLR_debug/'
			# else:
			# 	args.save_model_dir = ''

			fix_random_seed(int(args.seed))
			trainer = Trainer(hparams, dataset_configs, device, args)
			trainer.train()
	else:
		trainer = Trainer(hparams, dataset_configs, device, args)
		trainer.load_model(args.ckpt)
		trainer.test()
		if args.plotdir == '':
			trainer.save_plot(args.plotdir, args.ckpt.split('/')[1][:-4])






