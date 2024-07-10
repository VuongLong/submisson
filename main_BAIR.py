import argparse
import json
import os
import random

import numpy as np
import torch
from BAIR.src.Trainer import Trainer


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


algorithms_map = {"KEFI": Trainer}


def get_args():
	parser = argparse.ArgumentParser()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config", help="Path to configuration file")
	parser.add_argument("--gpu_idx", help="Index of GPU")
	parser.add_argument("--exp_idx", type=int, default=0, help="Index of experiment")

	# Prototype
	parser.add_argument("--prototype_per_class", type=int, default=16, help="4 8 16 32")
	
	# Invariant Prediction constraint
	parser.add_argument("--env_weight",type=float, default=1.0, help="Experiment configs")
	parser.add_argument("--is_bias", type=int, default=0, help="Experiment configs")

	# Feature regularizarion: spread out to increase information
	parser.add_argument("--pcl_weight",type=float, default=0.1, help="Experiment configs")

	# OT Regularization between feature and prototype
	parser.add_argument("--ot_weight",type=float, default=0.1, help="Experiment configs")
	parser.add_argument("--ot_scale",type=float, default=0.1, help="Experiment configs")


	parser.add_argument("--disc_weight",type=float, default=0.01, help="Experiment configs")
	parser.add_argument("--disc_threshold",type=float, default=10, help="Experiment configs")
	parser.add_argument("--disc_cross",type=int, default=0, help="Experiment configs")
	parser.add_argument("--entropy",type=int, default=1, help="Experiment configs")

	# swad: average moving model
	parser.add_argument("--start_swad",type=float, default=0.0, help="Experiment configs")

	parser.add_argument("--save_model_dir",type=str, default='', help="Experiment configs")
	parser.add_argument("--algorithm", type=str, default="KEFI", help="Experiment configs")

	parser.add_argument("--step_eval", type=int, default=300, help="step to eval")
	parser.add_argument("--batch_size",type=int, default=32, help="")
	
	parser.add_argument("--ckpt",type=str, default='', help="Experiment configs")
	parser.add_argument("--plot_dir",type=str, default='', help="Experiment configs")
	parser.add_argument("--model_name",type=str, default='', help="Experiment configs")

	bash_args = parser.parse_args()
	with open(bash_args.config, "r") as inp:
		args = argparse.Namespace(**json.load(inp))

	# import pdb; pdb.set_trace()
	# os.environ["CUDA_VISIBLE_DEVICES"] = str(bash_args.gpu_idx)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.dataset == 'PACS':
		args.src_data_path = '/home/shared/data/DomainBed/PACS/'
	elif args.dataset == 'VLCS':
		args.src_data_path = '/home/shared/data/DomainBed/VLCS/'
	elif args.dataset == 'TerraInc': 
		args.src_data_path = '/home/shared/data/DomainBed/'
	elif args.dataset == 'OfficeHome':
		args.src_data_path = '/home/shared/data/DomainBed/'
		bash_args.prototype_per_class = 8
	else:
		args.src_data_path = '/home/shared/data/DomainBed/DomainNet/'

	# args.learning_rate = 0.00001
	exp_seed = [0, 1]
	args.batch_size = bash_args.batch_size
	args.step_eval = bash_args.step_eval

	print(args)
	print(bash_args)
	if bash_args.ckpt == '':

		for seed in exp_seed:
			bash_args.exp_idx = seed
			fix_random_seed(int(bash_args.exp_idx))
			trainer = algorithms_map[bash_args.algorithm](args, device, bash_args)
			trainer.train()
	else:
		trainer = algorithms_map[bash_args.algorithm](args, device, bash_args)
		trainer.load_model(bash_args.ckpt)
		if bash_args.plot_dir == '':
			trainer.test()
		else:
			trainer.save_plot(bash_args.plot_dir)






