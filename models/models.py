import torch
import torch.nn as nn
import torch.nn.functional as F

from typing      import Optional
from constants   import *


class PECT2DModel(nn.Module):
    """CheXpert using classic imagenet models"""
    def __init__(
            self, 
            model_name: str = 'densenet121', 
            num_classes: int = 1, 
            ckpt_path: Optional[str] = None, 
            imagenet_pretrain: bool = True
        ):
        super(PECT2DModel, self).__init__()

        self.model_name = model_name 
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.imagenet_pretrain = imagenet_pretrain

        # Check if using contrastive pretrained 
        model_2d, features_dim = MODELS_2D[model_name]
        self.model = model_2d(pretrained=self.imagenet_pretrain)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(features_dim, self.num_classes)

    def forward(self, x):
        if self.model_name.startswith(('resnet', 'resnext')):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

        elif self.model_name.startswith("dense"):
            x = self.model.features(x)
            x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.fc(x)
        return x

    def args_dict(self):
        ckpt = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "ckpt_path": self.ckpt_path,
            "imagenet_pretrain": self.imagenet_pretrain
        }
        return ckpt
