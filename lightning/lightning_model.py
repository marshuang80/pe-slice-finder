import sys
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import numpy as np
import utils
import wandb
sys.path.append(os.getcwd())

from models          import PECT2DModel
from eval            import *
from argparse        import ArgumentParser
from dataset         import get_dataloader
from pytorch_lightning.metrics.classification import AveragePrecision, AUROC
from sklearn.metrics        import f1_score, average_precision_score, roc_auc_score


class LightningModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #self.save_hyperparameters() TODO: wandb bug of storing hparam in hparam

        self.loss = get_loss_fn(self.hparams)
        self.model = PECT2DModel(
            model_name = self.hparams.model_name,
            imagenet_pretrain = self.hparams.imagenet_pretrain,
            ckpt_path = self.hparams.ckpt_path
        )

        # for computing metrics
        self.train_probs = []
        self.val_probs = []
        self.test_probs = []
        self.train_true = []
        self.val_true = []
        self.test_true = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.cuda.FloatTensor)
        y_hat = self.model(x)

        # compute loss
        y_hat = y_hat[:,0]
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # logging
        if batch_idx == 0:
            self.log_image(x)
        result = pl.TrainResult(loss)
        result.log(
            'train_loss', loss, on_epoch=True, on_step=True, 
            sync_dist=True, logger=True, prog_bar=True)
        self.train_probs.append(probs.cpu().detach().numpy())
        self.train_true.append(y.cpu().detach().numpy())

        return result

    def training_epoch_end(self, training_result):
        # log metric
        auroc, auprc = self.evaluate(self.train_probs, self.train_true)
        training_result.log('train_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        training_result.log('train_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        training_result.epoch_train_loss = torch.mean(training_result.train_loss)

        # reset 
        self.train_probs = []
        self.train_true = []
        return training_result
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.cuda.FloatTensor)
        y_hat = self.model(x)

        # compute loss
        y_hat = y_hat[:,0]
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # log loss
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.val_probs.append(probs.cpu().detach().numpy())
        self.val_true.append(y.cpu().detach().numpy())

        return result

    def validation_epoch_end(self, validation_result):
        # log metrics
        auroc, auprc = self.evaluate(self.val_probs, self.val_true)
        validation_result.log('val_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        validation_result.log('val_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        validation_result.val_loss = torch.mean(validation_result.val_loss)

        # reset 
        self.val_probs = []
        self.val_true = []
        return validation_result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.cuda.FloatTensor)
        y_hat = self.model(x)

        # compute loss
        y_hat = y_hat[:,0]
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # log loss
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.test_probs.append(probs.cpu().detach().numpy())
        self.test_true.append(y.cpu().detach().numpy())

        return result

    def test_epoch_end(self, test_result):
        # log metrics
        auroc, auprc = self.evaluate(self.test_probs, self.test_true)
        print(f"Test AUROC: {auroc}")
        print(f"Test AUPRC: {auprc}")

        test_result.log('test_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        test_result.log('test_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        test_result.test_loss = torch.mean(test_result.test_loss)

        # reset 
        self.test_probs = []
        self.test_true = []
        return test_result 


    def evaluate(self, probs, true):

        # concat results from all iterations
        probs = np.concatenate(probs)
        true = np.concatenate(true)

        # can't calculate metric if no positive example 
        if (1 not in true) or (0 not in true):
            auprc = 0
            auroc = 0
        else:
            auprc = average_precision_score(true, probs)
            auroc = roc_auc_score(true, probs)
        
        return auroc, auprc

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        else: 
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
    
    def log_image(self, x):
        image = x[0].cpu().numpy()
        image = np.transpose(image, (1,2,0))
        image = wandb.Image(image, caption='SampleImage')
        self.logger.experiment.log({'example': image}) 

    def __dataloader(self, split):

        transforms = utils.get_transformations(self.hparams, split)
        shuffle = split == "train"
        drop_last = split == "train"
        weighted_sampling = (split == "train") & self.hparams.weighted_sampling

        dataset_args = {
            'transforms': transforms, 
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
