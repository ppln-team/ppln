# model settings
model = 'resnet18'
sync_bn = True

# transform settings
pre_transforms = [
    dict(type='LongestMaxSize', max_size=64),
    dict(type='PadIfNeeded', min_height=64, min_width=64),
]
post_transforms = [dict(type='Normalize')]
augmentations = [
    dict(type='HorizontalFlip'),
    dict(type='ShiftScaleRotate', shift_limit=0.11, scale_limit=0.13, rotate_limit=7)
]
transforms = dict(train=pre_transforms + augmentations + post_transforms, val=pre_transforms + post_transforms)

# dataset settings
data = dict(
    data_root='/data/cifar10',
    images_per_gpu=64,  # images per gpu
    workers_per_gpu=4  # data workers per gpu
)

# apex settings
apex = dict(opt_level='O2', keep_batchnorm_fp32=True, loss_scale=512.0, delay_allreduce=True)

# optimizer and learning rate
optimizer = dict(type='torch.optim.Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    type='ReduceLROnPlateauHook',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.33,
    metric_name='loss',
    mode='max',
    patience=0
)

# runtime settings
work_dir = '/data/demo'
dist_params = dict(backend='nccl')
checkpoint_config = dict(num_checkpoints=5, metric_name='acc_top5', mode='max')  # save checkpoint at every epoch
total_epochs = 20
resume_from = None
load_from = None

# logging settings
log_config = dict(hooks=[dict(type='ProgressBarLoggerHook', bar_width=40), dict(type='TextLoggerHook')])
