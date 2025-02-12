import io
import math
import os
from itertools import permutations
from typing import OrderedDict

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import AVRModule


class ScoringModel(AVRModule):
    def __init__(
        self,
        cfg: DictConfig,
        context_norm: bool,
        num_correct: int,  # number of correct answers, raven - 1, bongard - 2
        in_dim,
        transformer: pl.LightningModule,
        pos_emb: pl.LightningModule | None = None,
        disc_pos_emb: pl.LightningModule | None = None,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        increment_dataloader_idx: int = 0,
        **kwargs,
    ):
        super().__init__(cfg)

        if save_hyperparameters:
            self.save_hyperparameters(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )

        for _key, _value in kwargs.items():
            self.__dict__[_key] = _value
        # self.automatic_optimization = False # use when slot and transoformer have separate optimizers
        self.in_dim = in_dim
        # self.row_column_fc = nn.Linear(6, in_dim)
        self.num_correct = num_correct

        if context_norm: # whether to appl context norm
            self.contextnorm = True
            self.gamma = nn.Parameter(torch.ones(in_dim))
            self.beta = nn.Parameter(torch.zeros(in_dim))
        else:
            self.contextnorm = False



        multi_transformers = [ # multi transformer models handling
            int(_it.removeprefix("transformer_"))
            for _it in kwargs.keys()
            if _it.startswith("transformer_")
        ]
        multi_pos_emb = [ #multi positional embedding handling
            int(_it.removeprefix("pos_emb_"))
            for _it in kwargs.keys()
            if _it.startswith("pos_emb_")
        ]
        assert len(multi_transformers) == len(
            multi_pos_emb
        ), "Number of transformers and positional embeddings should be equal"

        if len(multi_transformers) == 0:
            self.transformer = nn.ModuleList([transformer])
            self.pos_emb = nn.ModuleList([pos_emb])
        if len(multi_transformers) > 0:
            self.transformer = nn.ModuleList(
                [transformer]
                + [kwargs.get(f"transformer_{_ix}") for _ix in multi_transformers]
            )
            self.pos_emb = nn.ModuleList(
                [pos_emb] + [kwargs.get(f"pos_emb_{_ix}") for _ix in multi_pos_emb]
            )

        multi_disc_pos_emb = [
            int(_it.removeprefix("disc_pos_emb_"))
            for _it in kwargs.keys()
            if _it.startswith("disc_pos_emb_")
        ]
        if len(multi_disc_pos_emb) == 0:
            self.disc_pos_emb = nn.ModuleList([disc_pos_emb])
        if len(multi_disc_pos_emb) > 0:
            self.disc_pos_emb = nn.ModuleList(
                [disc_pos_emb] + [kwargs.get(f"disc_pos_emb_{_ix}") for _ix in multi_disc_pos_emb]
            )

        self.use_disc_pos_emb = pos_emb is None and not disc_pos_emb is None

        self.loss = instantiate(cfg.metrics.cross_entropy)
        self.val_losses = []
        self.increment_dataloader_idx = increment_dataloader_idx

        def create_module_dict(metrics_dict):
            return nn.ModuleDict(
                {
                    metric_nm: (
                        instantiate(metric_func)
                        if isinstance(metric_func, DictConfig)
                        else metric_func
                    )
                    for metric_nm, metric_func in metrics_dict.items()
                }
            )

        if len(multi_transformers) == 0:  # loading additional metrics for different tasks from configuration files
            self.additional_metrics = nn.ModuleList(
                [create_module_dict(additional_metrics)]
            )
        else:
            self.additional_metrics = nn.ModuleList(
                [create_module_dict(additional_metrics)]
                + [
                    create_module_dict(kwargs.get(f"additional_metrics_{_ix}"))
                    for _ix in multi_transformers
                ]
            )

    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        # print("seq, mean, var shape>>",z_seq.shape,z_mu.shape,z_sigma.shape)
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma.unsqueeze(0).unsqueeze(0)) + self.beta.unsqueeze(
            0
        ).unsqueeze(0)
        return z_seq

    def forward(self, given_panels, answer_panels, idx=0):
        pass

    # TODO: Separate optimizers for different modules
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        pass

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("train", batch, batch_idx, dataloader_idx)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("val", batch, batch_idx, dataloader_idx)
        self.val_losses.append(loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("test", batch, batch_idx, dataloader_idx)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_losses = torch.tensor(self.val_losses)
        val_loss = val_losses.mean()
        self.log(
            "val/loss",
            val_loss.to(self.device),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_losses.clear()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        **kwargs,
    ):
        """
        Load a model from a checkpoint file --- due to changes in model structure we have to rename weights for some parts to match current version.
        """
        _checkpoint = torch.load(checkpoint_path, map_location="cpu")

        transformer_weights = [
            k for k in _checkpoint["state_dict"].keys() if k.startswith("transformer.")
        ]
        new_state_dict = OrderedDict()

        if len([k for k in transformer_weights if k.startswith("transformer.0")]) == 0:
            # renaming transformer to tranformer.0 to match the model (same with pos_emb -> pos_emb.0)
            weight_names = _checkpoint["state_dict"].keys()
            for key in weight_names:
                if key.startswith("transformer."):
                    new_state_dict[key.replace("transformer.", "transformer.0.", 1)] = (
                        _checkpoint["state_dict"].get(key)
                    )
                elif key.startswith("pos_emb."):
                    new_state_dict[key.replace("pos_emb.", "pos_emb.0.", 1)] = (
                        _checkpoint["state_dict"].get(key)
                    )
                else:
                    new_state_dict[key] = _checkpoint["state_dict"].get(key)
            _checkpoint["state_dict"] = new_state_dict

        buffer = io.BytesIO()
        torch.save(_checkpoint, buffer)

        return super().load_from_checkpoint(
            io.BytesIO(buffer.getvalue()),
            map_location=map_location,
            hparams_file=hparams_file,
            strict=strict,
            **kwargs,
        )
