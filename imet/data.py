import os.path as osp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2


class ImetDataset(Dataset):
    def __init__(self, images_path, annotation_path, transform=None):
        self.annotation_path = annotation_path
        self.transform = transform
        self.images_path = images_path
        self.id2tag, self.id2culture = self.id2labels()
        self.ids = list(self.id2tag.keys())

    def id2labels(self):
        annotation = pd.read_csv(self.annotation_path, converters={'sub_tag_ids': eval, 'sub_culture_ids': eval})
        id2tag = dict(zip(annotation['id'], annotation['sub_tag_ids']))
        id2culture = dict(zip(annotation['id'], annotation['sub_culture_ids']))
        return id2tag, id2culture

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        path = osp.join(self.images_path, f"{image_id}.png")
        tag_labels = np.array(self.id2tag[image_id], dtype=np.int64)
        culture_labels = np.array(self.id2culture[image_id], dtype=np.int64)

        data = {'image': cv2.imread(path)[..., ::-1], 'tag_labels': tag_labels, 'culture_labels': culture_labels}
        if self.transform is not None:
            return self.transform(data)

        return data
