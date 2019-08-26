# Python lib
import sys
sys.path.append('/home/daniel/gitrepos/vision/references/detection')
import torch
from engine import train_one_epoch, evaluate, _get_iou_types
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
from glob import glob
from sklearn.model_selection import train_test_split
from coco_utils import get_coco_api_from_dataset
# from coco_eval import CocoEvaluator

from hparams import create_hparams

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_data(hparams):
    print("Batch size: ", hparams.batch_size)
    if hparams.checkpoint != None:
        label_encoder = torch.load(f"{hparams.checkoint_path}/label_encoder.bin")
    else:
        label_encoder = LabelEncoder()
        train_df = pd.read_csv(f"{hparams.csv_dir}/train-annotations-bbox.csv")
        train_df["LabelEncoded"] = label_encoder.fit_transform(train_df["LabelName"])
        print("Finished encoding labels")
        torch.save(label_encoder,f"{hparams.checkpoint_path}/label_encoder.bin")

    collapsed_train_df = pd.read_csv(f"{hparams.csv_dir}/train-annotations-bbox-collapsed.csv")

    uncollapsed_train_df = pd.read_csv(f"{hparams.csv_dir}/train-annotations-bbox.csv")
    num_classes = uncollapsed_train_df["LabelName"].nunique()

    # y = collapsed_train_df.pop('LabelName')
    # x = collapsed_train_df
    # train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.01, random_state=42)
    # train_x["LabelName"] = train_y
    # val_x["LabelName"] = val_y
    # train_df = train_x
    # val_df = val_x

    train_df, val_df= train_test_split(collapsed_train_df, test_size=0.01, random_state=42)

    print(f"Train length: {len(train_df)} Val length: {len(val_df)}")

    dataset_train = train_utils.OpenDataset(hparams.train_images_dir, train_df, hparams.image_size, hparams.image_size, label_encoder, img_transforms=None, dataset_size=hparams.dataset_size)
    dataset_val = train_utils.OpenDataset(hparams.train_images_dir, val_df, hparams.image_size, hparams.image_size, label_encoder, img_transforms=None,dataset_size=hparams.dataset_size)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=hparams.batch_size, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)


    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=hparams.batch_size, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)


    return train_loader, val_loader, label_encoder, num_classes
    # return train_loader, label_encoder, num_classes

def MAP():
    return

def inference(data_loader, model):
    ''' Returns predictions and targets, if any. '''
    model.eval()

    all_predicts, all_targets = [], []

    print("Data loader length", len(data_loader))
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, disable=IN_KERNEL)):
            # if data_loader.dataset.mode != 'test':
            #     input_, target = data
            # else:
            #     input_, target = data, None
            images, targets = data

            output = model(images.cuda())



            all_predicts.append(output)

            if target is not None:
                all_targets.append(targets)

    predicts = torch.cat(all_predicts)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, targets

def eval(val_loader, train_loader, model, tensorboard, epoch):
    predicts_gpu, targets_gpu = inference(val_loader, model)
    val_map = MAP(predicts_gap_gpu, targets_gpu)
    # num_correct = torch.sum(predicts_gap_gpu.cpu() == targets_gpu.cpu())
    predicts, confs, targets = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy(), targets_gpu.cpu().numpy()


    labels = [label_encoder.inverse_transform(pred) for pred in predicts]

    assert len(labels) == len(val_loader.dataset.df)

    print(f"Val GAP: {val_gap}, Num correct: {num_correct}")

    tensorboard.log_scalar("val_num_correct", num_correct, epoch)
    tensorboard.log_scalar("val_gap", val_gap, epoch)

    val_df = val_loader.dataset.df
    train_df = train_loader.dataset.df
    rand_idx = int(random() * len(val_df))

    sample_row = val_df.iloc[rand_idx]
    sample_target = sample_row["landmark_id"]
    sample_prediction = int(predicts[rand_idx])

    sample_predict_image_name = train_df[train_df["landmark_id"] == sample_prediction]["id"].tolist()[0]
    sample_predict_image_path = f"/home/daniel/kaggle/landmarks/all_images_resized_{ORIGINAL_IMAGE_SIZE}/{sample_predict_image_name}.jpg"
    sample_correct_label_image_path = f"/home/daniel/kaggle/landmarks/all_images_resized_{ORIGINAL_IMAGE_SIZE}/{sample_row['id']}.jpg"

    images = [image_append_text(sample_predict_image_path,sample_prediction), image_append_text(sample_correct_label_image_path,sample_target)]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    tensorboard.log_image("predicted_and_target_image", np.asarray(new_im), epoch)

def load_model(num_classes, device, hparams):
    model_ft = train_utils.get_instance_segmentation_model(num_classes)
    if hparams.multi_gpu:
        print("[Using all the available GPUs]")
        model_ft = nn.DataParallel(model_ft, device_ids=[0, 1])

    model_ft.to(device)

    params = [p for p in model_ft.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)

    return model_ft, optimizer, lr_scheduler

if __name__ == "__main__":
    hparams = create_hparams()

    tensorboard = train_utils.Tensorboard(hparams.checkpoint_path + "/logdir")
    train_utils.prepare_checkpoint(hparams)
    device = torch.device(f'cuda:{hparams.gpu_id}')

    print("Loading data...")
    train_loader, val_loader, label_encoder, num_classes = load_data(hparams)
    # train_loader, label_encoder, num_classes = load_data(hparams)
    print("Finished loading data")
    print("Num classes: ", num_classes)
    model_ft, optimizer, lr_scheduler = load_model(num_classes, device, hparams)

    # Prepare coco datasets
    # train_coco = get_coco_api_from_dataset(train_loader.dataset)
    # iou_types = _get_iou_types(model_ft)
    # train_coco_evaluator = CocoEvaluator(train_coco, iou_types)

    val_coco = get_coco_api_from_dataset(val_loader.dataset)

    # val_coco_evaluator = CocoEvaluator(val_coco, iou_types)

    num_epochs = 8
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        avg_loss = train_one_epoch(model_ft, optimizer, train_loader, device, epoch, tensorboard, print_freq=1000)


        evaluate(model_ft, val_loader, device, tensorboard, val_coco, epoch)

        lr_scheduler.step()
        # torch.save(model_ft.state_dict(), f"{hparams.checkpoint_path}/checkpoint_{epoch}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_ft.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            # 'map': avg_map,
            'label_encoder': label_encoder,
            'image_size': hparams.image_size
        }, f"{hparams.checkpoint_path}/checkpoint_{epoch}")

        if len(glob(f"{hparams.checkpoint_path}/*")) - 1 > hparams.max_checkpoints_to_keep:
            os.remove(f"{hparams.checkpoint_path}/checkpoint_{epoch-hparams.max_checkpoints_to_keep}")
