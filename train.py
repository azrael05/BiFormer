import os
import argparse
import torch
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from models.biformer import BiFormer
from my_dataset import MyDataSet
#from model import swin_tiny_patch4_window7_224 as create_model
from utils_copy import read_split_data, read_data,train_one_epoch, evaluate
import timm
import cv2
import numpy as np
import imutils
from PIL import Image
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    RandomAffine,
                                    ToTensor)
# import sys
# sys.path.insert(0,"E:\BiFormer")
import pathlib   
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath
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


def save_best_model(model, best_acc, current_acc, epoch):
    """Compare the accuracy and save the model if improvemnet
    :param model: the model
    :type model: torch.nn
    :param best_acc: current best accuracy
    :type best_acc: float
    :param current_acc: current accuracy
    :type current_acc: float
    :param epoch: can be used to in the name of the model. Currently not used in naming
    :type epoch: int
    """
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), "./weights/best_model.pth")
    return best_acc


def main(args):
    """main function to run the training code
    """
    ## Check if gpu is present
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    ## Load the images paths and labels for train and validation
    train_images_path, train_images_label, val_images_path, val_images_label = read_data(args.data_path)
    img_size = 224
    shear_degrees = 10

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

    train_dataset = MyDataSet(images_path=train_images_path[:10],
                              images_class=train_images_label[:10],
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path[:10],
                            images_class=val_images_label[:10],
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    #model = timm.create_model('deit_small_patch16_224', pretrained=True,num_classes=args.num_classes).to(device)
    model_urls = {
    "biformer_tiny_in1k": "https://matix.li/e36fe9fb086c",
    "biformer_small_in1k": "https://matix.li/5bb436318902",
    "biformer_base_in1k": "https://matix.li/995db75f585d",
    }

    ## Current architecture set for ourcase can be modified or imported from model.biformer file 
    model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        num_classes=2,
        #------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None).to(device)
        #-------------------------------
#         **kwargs)

    ## Load the tiny weights from hugging face
    model_key = 'biformer_tiny_in1k'
    url = model_urls[model_key]
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", file_name="biformer_tiny_in1k.pth")

    for k in list(checkpoint["model"].keys()):
        if "head" in k:
            del checkpoint["model"][k]
    model.load_state_dict(checkpoint["model"],strict=False)

    ## If weights file provided in argument load from that
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    ## if args.freeze =True then freeze all weights except the final head layer
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    best_val_acc = 0.0
    
    ## Training of model
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        best_val_acc = save_best_model(model, best_val_acc, val_acc, epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)