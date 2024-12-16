import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class Sequential(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        models: list[nn.Module],
    ):
        super().__init__()
        self.model = nn.Sequential(*models)

    def forward(self, x):
        return self.model(x)
