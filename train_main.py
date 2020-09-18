from argparse import ArgumentParser
from args import BaseArgParser
from lightning import LightningModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping


seed_everything(6)

def main(args):

    # logger
    logger = pl_loggers.WandbLogger(
        name=None,
        save_dir=None,
        experiment=None 
    )

    # early stop call back
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        strict=False,
        verbose=False,
        mode='min'
    )

    model = LightningModel(args)
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        early_stop_callback=early_stop
        )
    trainer.fit(model)


if __name__ == '__main__':
    parser = BaseArgParser().get_parser()

    # add model specific args
    parser = LightningModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)