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

from utils import progress_bar
from loss.spc import SupervisedContrastiveLoss
from data_augmentation.auto_augment import AutoAugment
from data_augmentation.duplicate_sample_transform import DuplicateSampleTransform

# from models.resnet_contrastive import get_resnet_contrastive
from models_copy.biformer import BiFormer
# from models.biformer import BiFormer
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

def parse_args():
    parser = argparse.ArgumentParser()



    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument("--temperature", default=0.07, type=float, help="Constant for loss no thorough ")
    parser.add_argument("--auto-augment", default=False, type=bool)

    parser.add_argument("--n_epochs_contrastive", default=1, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=1, type=int)

    parser.add_argument("--lr_contrastive", default=1e-1, type=float)
    parser.add_argument("--lr_cross_entropy", default=5e-2, type=float)

    parser.add_argument("--cosine", default=True, type=bool, help="Check this to use cosine annealing instead of ")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")
    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=[150, 300, 500],
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    parser.add_argument("--num_workers", default=1, type=int, help="number of workers for Dataloader")

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, mode, args):
    """

    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    if mode == "contrastive":
        lr = args.lr_contrastive
        n_epochs = args.n_epochs_contrastive
    elif mode == "cross_entropy":
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
    else:
        raise ValueError("Mode %s unknown" % mode)

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
    else:
        n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if n_steps_passed > 0:
            lr = lr * (args.lr_decay_rate ** n_steps_passed)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def     train_contrastive(model, train_loader, criterion, optimizer, writer, args):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return: None
    """
    model.train()
    best_loss = float("inf")

    for epoch in range(args.n_epochs_contrastive):
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_contrastive))

        train_loss = 0
        for batch_idx, (inputsp, targets) in enumerate(train_loader):
            # print(inputs)
            inputs = torch.cat(inputsp)
            targets = targets.repeat(2)

            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            projections = model.forward_constrative(inputs)
            loss = criterion(projections, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar(
                "Loss train | Supervised Contrastive",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f " % (train_loss / (batch_idx + 1)),
            )
            del inputs, targets
        torch.cuda.empty_cache()
        gc.collect()

        avg_loss = train_loss / (batch_idx + 1)
        # Only check every 10 epochs otherwise you will always save
    
        if (train_loss / (batch_idx + 1)) < best_loss:
            print("Saving..")
            state = {
                "net": model.state_dict(),
                "avg_loss": avg_loss,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt_contrastive.pth")
            best_loss = avg_loss

        adjust_learning_rate(optimizer, epoch, mode="contrastive", args=args)
    


def train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = np.array(inputs)
            print(inputs.shape)
            targets = np.array(targets)

            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total_batch = targets.size(0)
            correct_batch = predicted.eq(targets).sum().item()
            total += total_batch
            correct += correct_batch

            writer.add_scalar(
                "Loss train | Cross Entropy",
                loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            writer.add_scalar(
                "Accuracy train | Cross Entropy",
                correct_batch / total_batch,
                epoch * len(train_loader) + batch_idx,
            )
            progress_bar(
                batch_idx,
                len(train_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

        validation(epoch, model, test_loader, criterion, writer, args)

        adjust_learning_rate(optimizer, epoch, mode='cross_entropy', args=args)
    print("Finished Training")


def validation(epoch, model, test_loader, criterion, writer, args):
    """

    :param epoch: int
    :param model: torch.nn.Module, Model
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module, Loss
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, "./checkpoint/ckpt_cross_entropy.pth")
        args.best_acc = acc


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # if args.dataset == "cifar10":
    #     mean = (0.4914, 0.4822, 0.4465)
    #     std = (0.2023, 0.1994, 0.2010)
    #     transform_train = [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #     ]
    #     if args.auto_augment:
    #         transform_train.append(AutoAugment())
    #     transform_train.extend(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean, std),
    #         ]
    #     )
    #     transform_train = transforms.Compose(transform_train)

    #     transform_test = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean, std),
    #         ]
    #     )
    # import sys
    # sys.path.insert(0,"E:\BiFormer")
    from my_dataset import MyDataSet
    from utils_copy import read_data
    # from train_copy import rotate_crop
    from torchvision.transforms import (CenterCrop,
                                Compose,
                                Normalize,
                                RandomHorizontalFlip,
                                RandomResizedCrop,
                                Resize,
                                RandomAffine,
                                ToTensor)
    img_size = 224
    shear_degrees = 10
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
]),}
    train_contrastive_transform = DuplicateSampleTransform(data_transform["train"])
    train_contrastive_transform_val = DuplicateSampleTransform(data_transform["val"])
    train_images_path, train_images_label, val_images_path, val_images_label = read_data("/kaggle/working/cifake")
    # Zip the two lists together
    import random
    zipped_lists = list(zip(train_images_path[:10000], train_images_label[:10000]))
    # Shuffle the zipped list
    random.shuffle(zipped_lists)
    # Unzip the shuffled list
    train_images_path, train_images_label = zip(*zipped_lists)
    zipped_lists = list(zip(val_images_path[:10000], val_images_label[:10000]))
    # Shuffle the zipped list
    random.shuffle(zipped_lists)
    # Unzip the shuffled list
    val_images_path, val_images_label = zip(*zipped_lists)
    
    train_dataset = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            transform=train_contrastive_transform)

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=train_contrastive_transform_val)
    # train_set = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=args.num_workers,
    )

    # test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
    )
    train_images_path, train_images_label, val_images_path, val_images_label = read_data("/kaggle/working/cifake")
    train_dataset_norm = MyDataSet(images_path=train_images_path,
                            images_class=train_images_label,
                            transform=data_transform["train"])

    val_dataset_norm = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    # train_set = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform_train)
    train_loader_norm = torch.utils.data.DataLoader(
        train_dataset_norm,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=args.num_workers,
    )

    # test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform_test)
    test_loader_norm = torch.utils.data.DataLoader(
        val_dataset_norm,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=args.num_workers,
    )
    num_classes = 2

    model = BiFormer(
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
        pe=None).to(device)
    # model = model.to(args.device)

    cudnn.benchmark = True

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    writer = SummaryWriter("logs")

    # if args.training_mode == "contrastive":
    #     train_contrastive_transform = DuplicateSampleTransform(transform_train)
        # if args.dataset == "cifar10":
        #     train_set_contrastive = datasets.CIFAR10(
        #         root="~/data",
        #         train=True,
        #         download=True,
        #         transform=train_contrastive_transform,
        #     )
        # elif args.dataset == "cifar100":
        #     train_set_contrastive = datasets.CIFAR100(
        #         root="~/data",
        #         train=True,
        #         download=True,
        #         transform=train_contrastive_transform,
        #     )

        # train_loader_contrastive = torch.utils.data.DataLoader(
        #     train_set_contrastive,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     drop_last=True,
        #     num_workers=args.num_workers,
        # )

        # model = model.to(args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_contrastive,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    criterion.to(args.device)
    train_contrastive(model, train_loader, criterion, optimizer, writer, args)

    # Load checkpoint.
    # print("==> Resuming from checkpoint..")
    # assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    # checkpoint = torch.load("./checkpoint/ckpt_contrastive.pth")
    # model.load_state_dict(checkpoint["net"])

    model.freeze_projection()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_cross_entropy,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)
    
    torch.cuda.empty_cache()
    gc.collect()
    args.best_acc = 0.0
    train_cross_entropy(model, train_loader_norm, test_loader_norm, criterion, optimizer, writer, args)
    torch.cuda.empty_cache()
    gc.collect()
    # else:
    #     optimizer = optim.SGD(
    #         model.parameters(),
    #         lr=args.lr_cross_entropy,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay,
    #     )
    #     criterion = nn.CrossEntropyLoss()
    #     criterion.to(args.device)

    #     args.best_acc = 0.0
    #     train_cross_entropy(model, train_loader, test_loader, criterion, optimizer, writer, args)

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
    model.load_state_dict(torch.load(r'checkpoint\ckpt_contrastive.pth', map_location=torch.device("cpu")))
    img_size = 224
    data_transform = {
    ## Transformation to be applied on validation
    "val": transforms.Compose([
        # transforms.Lambda(rotate_crop),  # Apply rotate_crop before other transformations
        # transforms.Lambda(rotate_crop),
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
        for label in pred_classes:
            preds.append(label)
        # accu_num += torch.eq(pred_classes, labels.to("cpu")).sum()

    # print(np.sum(np.array(preds)==np.array(acts))/len(image_paths))
    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(acts, preds))
    print(confusion_matrix(acts, preds))
    
    del model

if __name__ == "__main__":
    # main()
    # import shutil
    # # shutil.rmtree(r"embeddings\contrastive")
    # # os.makedirs(r"embeddings\contrastive\REAL")
    # # os.makedirs(r"embeddings\contrastive\FAKE")
    
    from glob import glob
    image_paths = glob("..\\cifake\\val\\**\\**")[:100]
    # image_paths_2 = glob("..\\cifake\\val\\REAL\\**")[:100]
    inference(image_paths)
    inference(image_paths_2)
