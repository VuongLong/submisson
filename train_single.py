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
	parser.add_argument("--data_dir", help="data path")
	parser.add_argument("--dataset", help="dataset")
	parser.add_argument("--target", help="target domain")
	parser.add_argument("--seed", type=int, default=0, help="Index of experiment")

	# swad: average moving model
	parser.add_argument("--start_swad", type=float, default=0.0, help="Experiment configs")
	parser.add_argument("--save_model_dir", type=str, default='NIPS_checkpoint', help="Experiment configs")
	parser.add_argument("--ckpt", type=str, default='', help="Experiment configs")
	parser.add_argument("--plot_dir", type=str, default='', help="Experiment configs")


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	args = parser.parse_args()
	with open('domainbed/configs/{}_{}.json'.format(args.dataset, args.target), "r") as inp:
		dataset_configs = argparse.Namespace(**json.load(inp))
	dataset_configs.src_data_path = str(args.data_dir) + dataset_configs.dataset + '/'

	# setup hparams
	hparams = hparams_registry.default_hparams(dataset_configs.algorithm, dataset_configs.dataset)

	print(dataset_configs)
	print(hparams)
	
	if args.ckpt == '':
		fix_random_seed(int(args.seed))
		trainer = Trainer(hparams, dataset_configs, device, args)
		trainer.train()
	else:
		trainer = Trainer(hparams, dataset_configs, device, args)
		trainer.load_model(args.ckpt)
		if args.plot_dir == '':
			trainer.test()
		else:
			trainer.save_plot(args.plot_dir)






