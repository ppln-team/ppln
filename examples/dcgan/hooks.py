import torch
from torchvision.utils import make_grid

from ppln.hooks import HOOKS, TensorboardLoggerHook
from ppln.utils.misc import master_only


@HOOKS.register_module
class GANTensorboardLoggerHook(TensorboardLoggerHook):
    def __init__(self, log_dir, n_latent, n_images=8):
        super().__init__(log_dir)
        self.noise = torch.randn(n_images, n_latent, 1, 1, device='cuda')

    @master_only
    def log(self, runner):
        super().log(runner)
        sample_images = runner.model(self.noise, mode='G')
        self.writer.add_image('images', make_grid(sample_images, normalize=True), 0)
