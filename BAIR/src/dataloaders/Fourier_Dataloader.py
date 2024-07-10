from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import cv2
import os
import torch
import torch.nn.functional as F

from torchvision import transforms
import random
import torch
import numpy as np
from math import sqrt
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

class FourierDGDataset(Dataset):
    def __init__(self, src_path, sample_paths, class_labels, domain_label=-1, alpha=1.0):
        # self.transformer = transforms.Compose(
        #     [
        #         # transforms.Resize((224,224)),
        #         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.5),
        #         #transforms.RandomGrayscale(),
        #         #transforms.ToTensor(),
        #         #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )

        self.transformer = get_pre_transform(image_size=224, crop=True, jitter=0.4)
        self.names = sample_paths
        self.labels = class_labels
        self.post_transform = get_post_transform()
        self.from_domain = 'all'
        self.alpha = alpha

        self.src_path = src_path
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(self.names)):
            self.flat_names += self.names[i].tolist() 
            self.flat_labels += self.labels[i].tolist() 
            self.flat_domains += [i] * len(self.names[i].tolist())
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

        

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        
        img_name = os.path.join(self.src_path + img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)

        img_s, label_s, domain_s = self.sample_image(domain)
        img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
        img = [img_o, img_s, img_s2o, img_o2s]
        label = [label, label_s, label, label_s]
        domain = [domain, domain_s, domain, domain_s]

        return img, label, domain

    def sample_image(self, domain):
        if self.from_domain == 'all':
            domain_idx = random.randint(0, len(self.names)-1)
        elif self.from_domain == 'inter':
            domains = list(range(len(self.names)))
            domains.remove(domain)
            domain_idx = random.sample(domains, 1)[0]
        elif self.from_domain == 'intra':
            domain_idx = domain
        else:
            raise ValueError("Not implemented")
        img_idx = random.randint(0, len(self.names[domain_idx])-1)
        img_name_sampled = self.names[domain_idx][img_idx]
        
        img_name_sampled = os.path.join(self.src_path + img_name_sampled)
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        
        label_sampled = self.labels[domain_idx][img_idx]
        return self.transformer(img_sampled), label_sampled, domain_idx



def get_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path)
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    return DGDataset(args, names, labels, img_transform)





def dataset_info(filepath):
    with open(filepath, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_img_transform(train=False, image_size=224, crop=False, jitter=0):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        if crop:
            img_transform = [transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
        else:
            img_transform = [transforms.Resize((image_size, image_size))]
        if jitter > 0:
            img_transform.append(transforms.ColorJitter(brightness=jitter,
                                                        contrast=jitter,
                                                        saturation=jitter,
                                                        hue=min(0.5, jitter)))
        img_transform += [transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)]
        img_transform = transforms.Compose(img_transform)
    else:
        img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return img_transform


def get_pre_transform(image_size=224, crop=False, jitter=0):
    if crop:
        img_transform = [transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
    else:
        img_transform = [transforms.Resize((image_size, image_size))]
    if jitter > 0:
        img_transform.append(transforms.ColorJitter(brightness=jitter,
                                                    contrast=jitter,
                                                    saturation=jitter,
                                                    hue=min(0.5, jitter)))
    img_transform += [transforms.RandomHorizontalFlip(), lambda x: np.asarray(x)]
    img_transform = transforms.Compose(img_transform)
    return img_transform


def get_post_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform

def get_digit_transform(image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform


def get_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha

def get_centralized_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12
