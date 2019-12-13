from argparse import ArgumentParser

from cifar.experiment import CIFARBatchProcessor, CIFARExperiment
from ppln.runner import Runner
from ppln.utils.config import Config
from ppln.utils.misc import init_dist


def parse_args():
    parser = ArgumentParser(description='Train CIFAR-10 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_dist(**cfg.dist_params)

    experiment = CIFARExperiment(cfg)

    runner = Runner(
        model=experiment.model,
        optimizer=experiment.optimizer,
        batch_processor=CIFARBatchProcessor(),
        hooks=experiment.hooks,
        work_dir=experiment.work_dir
    )

    runner.run(
        data_loaders={
            'train': experiment.dataloader('train'),
            'val': experiment.dataloader('val')
        },
        max_epochs=cfg.total_epochs,
        resume_from=cfg.resume_from,
        load_from=cfg.load_from
    )


if __name__ == '__main__':
    main()
