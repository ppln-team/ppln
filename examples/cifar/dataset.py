from torchvision import datasets
import torch
from ppln.data.datasets import cv2_loader


class CustomCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(img.transpose(2, 0, 1))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
