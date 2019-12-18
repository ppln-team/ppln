import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from cifar.dataset import CustomCIFAR10
from ppln.batch_processor import BaseBatchProcessor
from ppln.experiment import BaseExperiment
from ppln.factory import make_optimizer
from ppln.metrics.accuracy import accuracy
from ppln.utils.misc import get_dist_info


class CIFARExperiment(BaseExperiment):
    @property
    def optimizers(self):
        return {'base': make_optimizer(self.model, self.cfg.optimizer)}

    def dataset(self, mode):
        return CustomCIFAR10(root=self.cfg.data.data_root, train=mode == 'train', transform=self.transform(mode))

    def sampler(self, mode, dataset):
        rank, world_size = get_dist_info()
        return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=mode == 'train')


class CIFARBatchProcessor(BaseBatchProcessor):
    def test_step(self, model, batch, **kwargs):
        prediction = model(batch['image'].cuda())
        return dict(
            values=torch.argmax(prediction.float(), dim=1).cpu().numpy(),
            num_samples=batch['image'].size(0),
            index=batch['index'].numpy(),
            gt_label=batch['target'].numpy()
        )

    def train_step(self, model, batch, **kwargs):
        prediction = model(batch['image'].cuda())
        target = batch['target'].cuda()
        loss = F.cross_entropy(prediction, target)

        return dict(
            base_loss=loss,
            values=dict(loss=loss.item(), **accuracy(prediction, target, topk=(1, 5))),
            num_samples=batch['image'].size(0)
        )

    def val_step(self, model, batch, **kwargs):
        return self.train_step(model, batch, **kwargs)
