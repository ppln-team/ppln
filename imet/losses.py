import numpy as np
import torch

from torch import nn


class MultiLabelSoftMax(nn.Module):
    """
    From https://github.com/azat-d/inclusive-images-challenge
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, predictions: torch.Tensor, labels: np.array):
        assert len(predictions) == len(labels)

        all_classes = np.arange(predictions.size(1), dtype=np.int64)
        zero_label = torch.tensor([0]).to(predictions.device)

        loss = 0
        denominator = 0
        for prediction, positive_classes in zip(predictions, labels):
            negative_classes = np.setdiff1d(all_classes, positive_classes, assume_unique=True)

            negative_classes = torch.tensor(negative_classes).to(predictions.device)
            positive_classes = torch.tensor(positive_classes).to(predictions.device).unsqueeze(dim=1)

            for positive in positive_classes:
                indices = torch.cat((positive, negative_classes))
                loss = loss + self.loss(prediction[indices].unsqueeze(dim=0), zero_label)
                denominator += 1

        loss /= denominator

        return loss
