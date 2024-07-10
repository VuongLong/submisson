# BAIR: Best Attainable Invariant Representation for DG


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

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /home/shared/data/DomainBed/  --algorithm BAIR --pretrained
```

Experiment results are reported as a table. In the table, the row `SWAD` and `SWAD_prototype` indicate out-of-domain accuracy from BAIR and BAIR using prototype to make prediction respectively.

The row `SWAD (inD)` indicates in-domain validation accuracy.

Example results:
```
+----------------+--------------+---------+---------+---------+---------+
|   Selection    | art_painting | cartoon |  photo  |  sketch |   Avg.  |
+----------------+--------------+---------+---------+---------+---------+
|     oracle     |   87.065%    | 83.422% | 96.856% | 80.089% | 86.858% |
|      iid       |   87.065%    | 81.077% | 94.686% | 77.576% | 85.101% |
|      last      |   83.160%    | 83.422% | 96.183% | 75.954% | 84.680% |
|   last (inD)   |   96.390%    | 95.880% | 95.537% | 94.754% | 95.640% |
|   iid (inD)    |   97.331%    | 97.040% | 95.959% | 96.601% | 96.733% |
|      SWAD      |   89.445%    | 84.168% | 98.054% | 83.302% | 88.742% |
|   SWAD (inD)   |   97.783%    | 97.906% | 97.163% | 97.984% | 97.709% |
| SWAD_prototype |   90.055%    | 83.582% | 98.129% | 83.524% | 88.822% |
+----------------+--------------+---------+---------+---------+---------+
```
In this example, the DG performance of BAIR and BAIR-prototype for PACS dataset are 88.742% and 88.822%.

If you set `indomain_test` option to `True`, the validation set is splitted to validation and test sets,
and the `(inD)` keys become to indicate in-domain test accuracy.


'PACS' 'VLCS' 'OfficeHome' 'TerraIncognita' 'DomainNet'

### Reproduce the results of the paper

We provide the instructions to reproduce the main results of the paper, Table 1 and 2.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.

- PACS

```
python train_single.py --config domainbed/configs/PACS_art.json
```

```
python train_all.py PACS0 --dataset PACS --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/  --algorithm BAIR --pretrained
python train_all.py PACS1 --dataset PACS --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py PACS2 --dataset PACS --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/  --algorithm BAIR --pretrained
```

- VLCS

```
python train_all.py VLCS0 --dataset VLCS --deterministic --seed 0 --checkpoint_freq 300 --tolerance_ratio 0.2 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py VLCS1 --dataset VLCS --deterministic --seed 1 --checkpoint_freq 300 --tolerance_ratio 0.2 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py VLCS2 --dataset VLCS --deterministic --seed 2 --checkpoint_freq 300 --tolerance_ratio 0.2 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
```

- OfficeHome

```
python train_all.py OH0 --dataset OfficeHome --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py OH1 --dataset OfficeHome --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py OH2 --dataset OfficeHome --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
```

- TerraIncognita

```
python train_all.py TR0 --dataset TerraIncognita --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py TR1 --dataset TerraIncognita --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
python train_all.py TR2 --dataset TerraIncognita --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained
```

- DomainNet

```
python train_all.py DN0 --dataset DomainNet --deterministic --seed 0 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained --prototype_per_class 4
python train_all.py DN1 --dataset DomainNet --deterministic --seed 1 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained --prototype_per_class 4
python train_all.py DN2 --dataset DomainNet --deterministic --seed 2 --checkpoint_freq 300 --data_dir /home/shared/data/DomainBed/ --algorithm BAIR --pretrained --prototype_per_class 4
```

## License

This source code is released under the MIT license, included [here](./LICENSE).

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.
