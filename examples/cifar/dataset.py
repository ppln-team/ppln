from torchvision import datasets


class CustomCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        sample = {'image': img, 'index': index, 'target': target}
        if self.transform is not None:
            sample = self.transform(**sample)

        if self.target_transform is not None:
            sample['target'] = self.target_transform(sample['target'])

        return sample
