CUDA_VISIBLE_DEVICES=0 python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 300 --data_dir /home/long/data/DomainBed --algorithm ERM --pretrained
CUDA_VISIBLE_DEVICES=1 python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 300 --data_dir /home/long/data/DomainBed --algorithm ERM --pretrained
CUDA_VISIBLE_DEVICES=2 python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 300 --data_dir /home/long/data/DomainBed --algorithm ERM --pretrained
