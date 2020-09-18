
import torch
import torch.nn.functional as F
from torch import nn as nn
from constants import *


class FocalLoss(nn.Module):
    """Focal loss function for imbalanced dataset. 
    Args:
        alpha (float): weighing factor between 0 and 1. Alpha may be set by inverse 
                       class frequency
        gamma (float):  modulating factor reduces the loss contribution from easy 
                        examples and extends the range in which an example receives 
                        low loss. Usually between 0 - 5.  
    """
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y):
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        pt = torch.exp(-bce_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

        
def get_loss_fn(args):

    if args.loss_fn == "BCE":
        if args.class_weight is not None:
            weight = torch.Tensor(args.class_weights) # 17.54
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss_fn == "FocalLoss":
        loss_fn = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    else:
        raise Exception(f'Sorry, {args.loss_fn} loss is not supported.')
    return loss_fn 

