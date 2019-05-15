# model settings
model = 'resnet18'
# dataset settings
data_root = '/data/cifar10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 256

# optimizer and learning rate
optimizer = dict(type='torch.optim.Adam', lr=3e-3)
optimizer_config = dict(grad_clip=None)
lr_config = dict(type='StepLrSchedulerHook', step=3)

# runtime settings
work_dir = '/data/demo'
gpus = range(1)
dist_params = dict(backend='nccl')
data_workers = 4  # data workers per gpu
checkpoint_config = dict(num_checkpoints=5, metric_name='acc_top5', mode='max')  # save checkpoint at every epoch
workflow = [('train', 1), ('val', 1)]
total_epochs = 20
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_config = dict(hooks=[dict(type='ProgressBarLoggerHook'), dict(type='TextLoggerHook')])
