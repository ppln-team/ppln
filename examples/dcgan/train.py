from argparse import ArgumentParser

from dcgan.batch_processor import GANBatchProcessor
from dcgan.experiment import GANExperiment

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

    experiment = GANExperiment(cfg)
    print(experiment.hooks)
    runner = Runner(
        model=experiment.model,
        optimizers=experiment.optimizers,
        batch_processor=GANBatchProcessor(cfg),
        hooks=experiment.hooks,
        work_dir=experiment.work_dir
    )

    runner.run(
        data_loaders={'train': experiment.dataloader('train')},
        max_epochs=cfg.total_epochs,
        resume_from=cfg.resume_from,
        load_from=cfg.load_from
    )


if __name__ == '__main__':
    main()
