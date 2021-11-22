import torch, torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import tqdm
from torch.utils.data import Dataset
import os
import csv
import kornia

import git
import zipfile
import os
import kaggle

class NIPS2017TargetedDataset(Dataset):
    def __init__(self, dataset_path):
        labels_csv_path = os.path.join(dataset_path, "images.csv")
        with open(labels_csv_path) as csvfile:
            labels_list = list(csv.reader(csvfile))[1:]

        self.image_names = [f"{row[0]}.png" for row in labels_list]
        image_paths = [
            os.path.join(dataset_path, "images", name) for name in self.image_names
        ]
        self.images = [PIL.Image.open(path).convert('RGB') for path in image_paths]
        self.true_classes = [int(row[6]) for row in labels_list]
        self.target_classes = [int(row[7]) for row in labels_list]

        categories_csv_path = os.path.join(dataset_path, "categories.csv")
        with open(categories_csv_path) as csvfile:
            categories_list = list(csv.reader(csvfile))[1:]
        self.class_names = [row[1] for row in categories_list]

        assert len(self.images) == len(self.true_classes) == len(self.target_classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, n):
        image_name = self.image_names[n][:-4]  # discard ".png"
        image_tensor = torchvision.transforms.functional.to_tensor(self.images[n])

        # since dataset has dummy class 0 and all labels are shifted
        true_class = self.true_classes[n] - 1
        target_class = self.target_classes[n] - 1

        true_class_name = self.class_names[true_class]
        target_class_name = self.class_names[target_class]

        return {
            "image": image_tensor,
            "image_name": image_name,
            "true_class": true_class,
            "target_class": target_class,
            "true_class_name": true_class_name,
            "target_class_name": target_class_name,
        }
    
def load_cifar_models():
    working_dir = os.path.abspath(os.getcwd())
    if not os.path.exists(os.path.join(working_dir, 'PyTorch_CIFAR10/')):
        git.Git(working_dir).clone("https://github.com/huyvnphan/PyTorch_CIFAR10.git")

    if not os.path.exists(os.path.join(working_dir, 'PyTorch_CIFAR10/cifar10_models/state_dicts')):
        from PyTorch_CIFAR10.data import CIFAR10Data
        CIFAR10Data.download_weights()
        if os.path.exists(os.path.join(working_dir, 'state_dicts.zip')):
            with zipfile.ZipFile(os.path.join(working_dir, "state_dicts.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(working_dir, "PyTorch_CIFAR10/cifar10_models/"))
        del CIFAR10Data
        
def load_nips_data():
    # First, load kaggle.json to /root/.kaggle via the command:
    # !mkdir ~/.kaggle
    # !cp /workspace/integrated_stAdv/kaggle.json ~/.kaggle/kaggle.json
    # !chmod 600 /root/.kaggle/kaggle.json
    working_dir = os.path.abspath(os.getcwd())
    if not os.path.exists(os.path.join(working_dir, "data/nips2017_targeted")):
        nips_path = os.path.join(working_dir, "data/nips2017_targeted")
        kaggle.api.dataset_download_files('google-brain/nips-2017-adversarial-learning-development-set', path=nips_path, unzip=True)
        print("NIPS2017 dataset downloaded")
        
