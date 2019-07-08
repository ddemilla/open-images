# Python lib
import sys
sys.path.append('/home/daniel/gitrepos/vision/references/detection')
import torch
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

import collections
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import train_utils

from hparams import create_hparams

hparams = create_hparams()

ImageFile.LOAD_TRUNCATED_IMAGES = True

if not os.path.exists(hparams.checkpoint_path):
    os.makedirs(hparams.checkpoint_path)

train_df = pd.read_csv(f"{hparams.csv_dir}/train-annotations-bbox.csv")

if hparams.checkpoint != None:
    label_encoder = torch.load(f"{hparams.checkoint_path}/label_encoder.bin")
else:
    label_encoder = LabelEncoder()
    train_df["LabelEncoded"] = label_encoder.fit_transform(train_df["LabelName"])
    print("Finished encoding labels")
    torch.save(label_encoder,f"{hparams.checkpoint_path}/label_encoder.bin")

collapsed_train_df = pd.read_csv(f"{hparams.csv_dir}/train-annotations-bbox-collapsed.csv")

train_csv = pd.read_csv(f"{hparams.csv_dir}/train-annotations-bbox.csv")
label_counts = train_csv["LabelName"].value_counts()
num_classes = len(label_counts)

device = torch.device('cuda:1')

train_images_dir = "data/unzipped/all_train"
train_csv_file = "data/csvs/train-annotations-bbox-collapsed.csv"
dataset_train = train_utils.OpenDataset(train_images_dir, train_csv_file, hparams.image_size, hparams.image_size, img_transforms=None)

model_ft = train_utils.get_instance_segmentation_model(num_classes)
if hparams.multi_gpu:
    print("[Using all the available GPUs]")
    model_ft = nn.DataParallel(model_ft, device_ids=[0, 1])
else:
    model_ft.to(device)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=hparams.batch_size, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)

params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
num_epochs = 8
for epoch in range(num_epochs):
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=1000)
    lr_scheduler.step()
    torch.save(model_ft.state_dict(), f"{hparams.checkpoint_path}/checkpoint_{epoch}")

    if len(glob(f"hparams.checkpoint_path/*")) - 1 > hparams.max_checkpoints_to_keep:
        os.remove(f"{hparams.checkpoint_path}/checkpoint_{epoch-hparams.max_checkpoints_to_keep}")