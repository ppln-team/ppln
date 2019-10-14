import os.path as osp

import cv2
import jpeg4py as jpeg
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def jpeg_loader(path):
    return jpeg.JPEG(path).decode()


def cv2_loader(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


class MultiLabelDataset(Dataset):
    def __init__(self, root, samples, num_classes, transform=None, target_transform=None, loader=jpeg_loader):
        self.root = root
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        result = {'image': self.loader(osp.join(self.root, sample['name'])), 'name': sample['name']}

        if 'classes' in sample:
            targets = sample['classes']
            if self.target_transform is not None:
                targets = [self.target_transform(target) for target in targets]
                targets = torch.tensor([float(cls in targets) for cls in range(self.num_classes)])
            result['targets'] = targets
        if self.transform is not None:
            result = self.transform(**result)
        return result


class ClassFolderDataset(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=jpeg_loader):
        super(ClassFolderDataset, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index):
        path, idx = self.samples[index]
        cls = self.classes[idx]
        image = self.loader(path)
        sample = {'image': image, 'class': cls, 'name': osp.basename(path)}
        if self.target_transform is not None:
            sample['target'] = self.target_transform(cls)
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample
