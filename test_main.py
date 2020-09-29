from argparse import ArgumentParser
from args import BaseArgParser
from lightning import LightningModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping


seed_everything(6)

def main(args):

    model = LightningModel.load_from_checkpoint(args.checkpoint_path)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model)


if __name__ == '__main__':
    parser = BaseArgParser().get_parser()
    parser.add_argument('--checkpoint_path', type=str)

    # add model specific args
    parser = LightningModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)