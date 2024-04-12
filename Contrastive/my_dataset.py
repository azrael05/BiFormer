from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """Load the dataset."""

    def __init__(self, images_path: list, images_class: list, transform=None):
        """Constructor image for MyDataset
        
        :param images_path: path of images
        :type images_path: list
        :param images_class: class of images
        :type images_class: list
        :param transform: type of transformation to be applied. Default None.
        :type transform: torch.transform
        """
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        """Returns number of images
        
        :retrun: Number of images as dataset length
        :rtype: int
        """
        return len(self.images_path)

    def __getitem__(self, item):
        """Get item from dataset
        
        :param item: number of image
        :type item: int
        :returns : Image, label
        :rtype: (PIL Image, str)
        """
        img = Image.open(self.images_path[item]).convert('RGB')
        
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        """Method for batching/stacking the images
        :param batch: batch_size
        :type batch: int
        :return: images, labels
        :type: list of images, list of labels
        """
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels