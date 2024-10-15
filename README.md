# Understanding Domain Generalization:\\A View of Necessity and Sufficiency


## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/home/shared/data/DomainBed/
```

### Environments

Environment details used for our study.

```
Python: 3.8.6
PyTorch: 1.7.0+cu92
Torchvision: 0.8.1+cu92
CUDA: 9.2
CUDNN: 7603
NumPy: 1.19.4
PIL: 8.0.1
```

## How to Run


`train_single.py` script conducts leave-one-out cross-validation for specific target domain.

```
python train_single.py --dataset PACS --target art_painting  --data_dir /home/shared/data/DomainBed/
```

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```
python train_all.py exp_name --dataset PACS --data_dir /home/shared/data/DomainBed/  --algorithm SRA
```

Experiment results are reported as a table. In the table, the row `classifer weight` and `prototype` indicate out-of-domain accuracy from SRA and SRA using prototype to make prediction respectively.


'PACS' 'VLCS' 'OfficeHome' 'TerraIncognita' 'DomainNet'

### Reproduce the results of the paper

We provide the instructions to reproduce the main results of the paper, Table 1 and 2.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.

- PACS

```
python train_all.py PACS0 --dataset PACS --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/  --algorithm SRA
python train_all.py PACS1 --dataset PACS --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py PACS2 --dataset PACS --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/  --algorithm SRA
```

- VLCS

```
python train_all.py VLCS0 --dataset VLCS --deterministic --seed 0 --checkpoint_freq 300 --tolerance_ratio 0.2 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py VLCS1 --dataset VLCS --deterministic --seed 1 --checkpoint_freq 300 --tolerance_ratio 0.2 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py VLCS2 --dataset VLCS --deterministic --seed 2 --checkpoint_freq 300 --tolerance_ratio 0.2 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
```

- OfficeHome

```
python train_all.py OH0 --dataset OfficeHome --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py OH1 --dataset OfficeHome --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py OH2 --dataset OfficeHome --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
```

- TerraIncognita

```
python train_all.py TR0 --dataset TerraIncognita --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py TR1 --dataset TerraIncognita --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py TR2 --dataset TerraIncognita --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
```

- DomainNet

```
python train_all.py DN0 --dataset DomainNet --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
python train_all.py DN1 --dataset DomainNet --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA 
python train_all.py DN2 --dataset DomainNet --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm SRA
```

## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.
