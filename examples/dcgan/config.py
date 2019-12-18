experiment_name = 'DCGAN'

n_latent = 100
# model settings
model = dict(type='dcgan.model.DCGAN', n_latent=n_latent, n_g_features=64, n_d_features=64, n_channels=3)

# dataset settings
data = dict(
    data_root='/data',
    images_per_gpu=128,  # images per gpu
    workers_per_gpu=4,  # data workers per gpu
    pin_memory=False,
    image_size=64
)

# ddp settings
dist_params = dict(backend='nccl')

# optimizer and learning rate
optimizer = dict(
    D=dict(type='torch.optim.Adam', lr=0.0002, betas=(0.5, 0.999)),
    G=dict(type='torch.optim.Adam', lr=0.0002, betas=(0.5, 0.999))
)

# runtime settings
work_dir = f'/data/dumps/{experiment_name}'
total_epochs = 10
resume_from = None
load_from = None

# hook settings
hooks = [
    dict(type='ProgressBarLoggerHook', bar_width=10),
    dict(type='TextLoggerHook'),
    dict(type='GANTensorboardLoggerHook', log_dir=work_dir, n_latent=n_latent),
    dict(type='ApexInitializeHook', opt_level='O1', loss_scale=128.0),
    dict(type='ApexDDPHook', delay_allreduce=True),
    dict(type='ApexOptimizerHook', grad_clip=None, name='G'),
    dict(type='ApexOptimizerHook', grad_clip=None, name='D')
]
