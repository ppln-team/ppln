import torch

from ppln.batch_processor import BaseBatchProcessor


class GANBatchProcessor(BaseBatchProcessor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def test_step(self, model, batch, **kwargs):
        pass

    def train_step(self, model, batch, **kwargs):
        data = batch[0].cuda()
        batch_size = data.size(0)
        real_label = torch.full((batch_size, ), 1, device=data.device)
        fake_label = torch.full((batch_size, ), 0, device=data.device)

        # generator
        noise = torch.randn(batch_size, self.cfg.n_latent, 1, 1, device=data.device)
        generated_data = model(noise, mode='G')
        g_loss = self.loss(model(generated_data, mode='D'), real_label)

        # discriminator
        real_loss = self.loss(model(data, mode='D'), real_label)

        fake_loss = self.loss(model(generated_data.detach(), mode='D'), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        return dict(
            G_loss=g_loss,
            D_loss=d_loss,
            values=dict(G_loss=g_loss.item(), D_loss=d_loss.item()),
            num_samples=batch_size
        )

    def val_step(self, model, data, **kwargs):
        self.train_step(model, data, **kwargs)
