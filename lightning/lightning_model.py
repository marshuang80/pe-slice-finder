import sys
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import utils
sys.path.append(os.getcwd())

from models          import PECT2DModel
from eval            import *
from argparse        import ArgumentParser
from dataset         import get_dataloader
from pytorch_lightning.metrics.classification import AveragePrecision, AUROC


class LightningModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hparams.tpu_cores = None  
        #self.save_hyperparameters() TODO: wandb bug of storing hparam in hparam

        self.transforms = utils.get_transformations(self.hparams)
        self.loss = get_loss_fn(hparams)
        self.model = PECT2DModel(
            model_name = hparams.model_name,
            imagenet_pretrain = hparams.imagenet_pretrain,
            ckpt_path = hparams.ckpt_path
        )

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.cuda.FloatTensor)
        y_hat = self.model(x)

        # metric 
        loss, auroc, auprc = self.evaluate(y, y_hat)

        # logging
        result = pl.TrainResult(loss)
        result.log('train_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        result.log('train_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        result.log('train_loss', loss, sync_dist=True, logger=True)
        result.log(
            'train_loss', loss, on_epoch=True, 
            on_step=False, sync_dist=True, logger=True
        )

        return result
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.cuda.FloatTensor)
        y_hat = self.model(x)

        loss, auroc, auprc = self.evaluate(y, y_hat) 

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('val_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        result.log('val_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)

        return result

    def evaluate(self, y, y_hat):
        y_hat = y_hat[:,0]
        probs = torch.sigmoid(y_hat)
        loss = self.loss(y_hat, y)

        # can't calculate metric if no positive example 
        if (1 not in y) or (0 not in y):
            auprc = torch.tensor(0.0).cuda()
            auroc = torch.tensor(0.0).cuda()
        else:
            auprc = AveragePrecision()(probs, y).cuda()
            auroc = AUROC()(probs, y).cuda()
        return loss, auroc, auprc

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        else: 
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)

    def __dataloader(self, split):
        shuffle = split == "train"
        drop_last = split == "train"
        weighted_sampling = (split == "train") & self.hparams.weighted_sampling
        dataset_args = {
            'transforms': self.transforms, 
            'normalize': self.hparams.normalize
        }
        dataloader_args = {
            'batch_size': self.hparams.batch_size,
            'num_workers': self.hparams.num_workers,
            'pin_memory': True, 
            'shuffle': shuffle, 
            'drop_last': drop_last
        }
        dataloader = get_dataloader(
            split=split,
            dataset_args=dataset_args,
            dataloader_args=dataloader_args,
            weighted_sampling=weighted_sampling
        )
        return dataloader
    
    def train_dataloader(self):
        dataloader = self.__dataloader("train")
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader("valid")
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader("test")
        return dataloader 

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='densenet121')
        parser.add_argument('--imagenet_pretrain', type=utils.str2bool, default=True)
        parser.add_argument('--ckpt_path', type=str, default=None)
        return parser
