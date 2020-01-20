import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

from ppln.experiment import BaseExperiment
from ppln.factory import make_optimizer, make_scheduler
from ppln.utils.misc import get_dist_info


class GANExperiment(BaseExperiment):
    @property
    def schedulers(self):
        return {
            'D': make_scheduler(self.model['D'], self.cfg.optimizer.D),
            'G': make_scheduler(self.model['G'], self.cfg.optimizer.G)
        }

    @property
    def optimizers(self):
        return {
            'D': make_optimizer(self.model['D'], self.cfg.optimizer.D),
            'G': make_optimizer(self.model['G'], self.cfg.optimizer.G)
        }

    def dataset(self, mode):
        return datasets.ImageFolder(root=self.cfg.data.data_root, transform=self.transform(mode))

    def transform(self, mode):
        return transforms.Compose(
            [
                transforms.Resize(self.cfg.data.image_size),
                transforms.CenterCrop(self.cfg.data.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def sampler(self, mode, dataset):
        rank, world_size = get_dist_info()
        return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=mode == 'train')
