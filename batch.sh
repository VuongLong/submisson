#!/bin/bash

#SBATCH --job-name=long-job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# SBATCH --nodelist=node02
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tung-long.vuong@monash.edu
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

source activate ../envs
set -x

python train_single.py --config domainbed/configs/${1}_${2}.json
