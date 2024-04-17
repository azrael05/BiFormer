import argparse
import math
import numpy as np
import os
import torch
import gc

from torch.backends import cudnn
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from Contrastive.utils import progress_bar
# from loss.spc import SupervisedContrastiveLoss
# from data_augmentation.auto_augment import AutoAugment
# from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform

# from models.resnet_contrastive import get_resnet_contrastive
# from models_copy.biformer import BiFormer
from models.biformer import BiFormer
import cv2
import imutils
from PIL import Image
def rotate_crop(image):
    """Rotate and crop the image
    :param image: the image
    :type image: PIL image
    """
    # Your existing rotate_crop function
    # ...
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    
    # convert the image to grayscale, blur it, and find edges
   
    (T, threshInv) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    masked = cv2.bitwise_and(image, image, mask=threshInv)
    blurred = cv2.GaussianBlur(threshInv, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 200)
    p1 = None# Adjust these values based on your points
    p2 = None
    edge_points = np.column_stack(np.where(edged == 255))
    for point in edge_points:
        x, y = point

        if p1 is None:
            p1 = point
        if p2 is None:
            p2 = point

        # Check if the current point is to the left of p1 or to the right of p2
        if y < p1[0]:
            p1 = point
        if y > p2[0]:
            p2 = point
    if p1 is None or p2 is None:
        return Image.fromarray(masked)
    angle = np.arctan2(p2[1]-p1[1], p2[0] - p1[0])
    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle)

    # Rotate the image
    rotated_image = imutils.rotate_bound(masked, -angle_degrees)
    rotated_image = Image.fromarray(rotated_image)
    return rotated_image


def inference(image_paths):
    from PIL import Image 
    model  = BiFormer(
         depth=[2, 2, 8],
        embed_dim=[64, 128, 256], mlp_ratios=[3, 3, 3],
        #------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1],
#         topks=[1, 4, 16, -2],
        topks = [1, -1,-2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None)
    model.load_state_dict(torch.load(r'weights\k_3\best_model.pth', map_location=torch.device("cpu")))
    img_size = 224
    shear_degrees = 10
    from torchvision. transforms import RandomAffine
    ## Transformation to be applied on training data
    data_transform = {
    "train": transforms.Compose([
        transforms.Lambda(rotate_crop),  # Apply rotate_crop before other transformations
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: transforms.functional.equalize(img)),
        RandomAffine(degrees=0, shear=shear_degrees),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),

    ## Transformation to be applied on validation
    "val": transforms.Compose([
        transforms.Lambda(rotate_crop),  # Apply rotate_crop before other transformations
        transforms.Resize(img_size),
        transforms.Lambda(rotate_crop),
        transforms.Lambda(lambda img: transforms.functional.equalize(img)),
        transforms.ToTensor(),
         transforms.Resize(img_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    }
    from my_dataset import MyDataSet
    from train import evaluate
    acts = []
    for image_path in image_paths:
        acts.append(0 if "FAKE" in image_path else 1)
    val_dataset = MyDataSet(images_path=image_paths,
                            images_class=acts,
                            transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=8,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    # val_loss, val_acc = evaluate(model=model,
    #                                  data_loader=val_loader,
    #                                  device="cpu",
    #                                  epoch=-1)
    # print(val_acc, val_loss)
    preds = []
    # from pathlib import Path
    from tqdm import tqdm
    for step, data in tqdm(enumerate(val_loader)):
        images, labels = data

        pred = model(images.to("cpu"))
        pred_classes = torch.max(pred, dim=1)[1]
        for label in labels:
            preds.append(label)
        # accu_num += torch.eq(pred_classes, labels.to("cpu")).sum()

    # print(np.sum(np.array(preds)==np.array(acts))/len(image_paths))
    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(acts, preds))
    print(confusion_matrix(acts, preds))
    
    del model

if __name__ == "__main__":
    # main()
    import shutil
    # shutil.rmtree(r"embeddings\contrastive")
    # os.makedirs(r"embeddings\contrastive\REAL")
    # os.makedirs(r"embeddings\contrastive\FAKE")
    
    from glob import glob
    image_paths = glob("cifake\\val\\**\\**")[:100]
    # image_paths_2 = glob("cifake\\val\\REAL\\**")
    # image_paths = image_paths + image_paths_2
    inference(image_paths)
    # inference(image_paths_2)
