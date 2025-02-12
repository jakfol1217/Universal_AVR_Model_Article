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
        slot_model: pl.LightningModule,
        transformer: pl.LightningModule,
        pos_emb: pl.LightningModule | None = None,
        disc_pos_emb: pl.LightningModule | None = None,
        additional_metrics: dict = {},
        save_hyperparameters=True,
        freeze_slot_model=True,
        auxiliary_loss_ratio: float = 0.0,
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

        self.slot_model = slot_model  
        if self.slot_model is not None:
            if ( # loading slot model weights from checkpoint
                slot_ckpt_path := cfg.model.slot_model.ckpt_path
            ) is not None and cfg.checkpoint_path is None:
                cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                model_cfg = {
                    k: v
                    for k, v in cfg_dict["model"]["slot_model"].items()
                    if k != "_target_"
                }
                self.slot_model = slot_model.__class__.load_from_checkpoint(
                    slot_ckpt_path, cfg=cfg, **model_cfg
                )
        self.freeze_slot_model = freeze_slot_model
        self.auxiliary_loss_ratio = auxiliary_loss_ratio # auxiliary loss ratio for image reconstruction with slots

        if self.slot_model is not None:   
            if self.freeze_slot_model:
                self.slot_model.freeze()
            else:
                self.slot_model.unfreeze()

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
        # computing scores with multiple possible transformer models
        __pos_emb = self.pos_emb[idx]
        __transformer = self.transformer[idx]
        __num_correct = (
            self.num_correct if idx == 0 else self.__dict__[f"num_correct_{idx}"]
        )
        __disc_pos_emb = self.disc_pos_emb[idx]

        scores = []
        pos_emb_score = (
            __pos_emb(given_panels) if __pos_emb is not None and not self.use_disc_pos_emb else torch.tensor(0.0)
        )

        disc_pos_embed = __disc_pos_emb() if self.use_disc_pos_emb else torch.rand(0)
        if self.use_disc_pos_emb:
            disc_pos_embed = disc_pos_embed.repeat(given_panels.shape[0], 1, 1)

        # Loop through all choices and compute scores
        for d in permutations(range(answer_panels.shape[1]), __num_correct):

            # print(AB,C_choices[:,d,:],AB.shape,C_choices[:,d,:].shape)
            # x_seq = torch.cat([given_panels_posencoded_seq,torch.cat((answer_panels[:,d],self.row_fc(third).unsqueeze(1).repeat((1,answer_panels.shape[2],1)), self.column_fc(third).unsqueeze(1).repeat((1,answer_panels.shape[2],1))),dim=2).unsqueeze(1)],dim=1)
            # print(given_panels.shape)
            # print(answer_panels[:, d].shape)
            x_seq = torch.cat([given_panels, answer_panels[:, d]], dim=1)
            # print("seq min and max>>",torch.min(x_seq),torch.max(x_seq))
            # x_seq = torch.cat([AB,C_choices[:,d,:].unsqueeze(1)],dim=1)
            x_seq = torch.flatten(x_seq, start_dim=1, end_dim=2)
            if self.contextnorm:

                x_seq = self.apply_context_norm(x_seq)

            if self.use_disc_pos_emb:
                x_seq = torch.cat([x_seq, disc_pos_embed], dim=-1)
            else:
                x_seq = x_seq + pos_emb_score  

            # x_seq = torch.cat((x_seq,all_posemb_concat_flatten),dim=2)
            score = __transformer(x_seq)
            scores.append(score)
        scores = torch.cat(scores, dim=1)
        return scores

    # TODO: Separate optimizers for different modules
    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch
        recon_combined_seq = []
        recons_seq = []
        masks_seq = []
        slots_seq = []
        for idx in range(img.shape[1]): # creating slots
            recon_combined, recons, masks, slots, attn = self.slot_model(img[:, idx])
            recons_seq.append(recons)
            recon_combined_seq.append(recon_combined)
            masks_seq.append(masks)
            slots_seq.append(slots)
            del recon_combined, recons, masks, slots, attn
        pred_img = torch.stack(recon_combined_seq, dim=1).contiguous()
        if pred_img.shape[2] != img.shape[2]:
            pred_img = pred_img.repeat(1, 1, 3, 1, 1)
        slot_model_loss = self.slot_model.loss(pred_img, img)

        context_panels_cnt = self.cfg.data.tasks[ # number of task context panels
            self.task_names[dataloader_idx]
        ].num_context_panels

        given_panels = torch.stack(slots_seq, dim=1)[:, :context_panels_cnt] # context panels
        answer_panels = torch.stack(slots_seq, dim=1)[:, context_panels_cnt:] # answer panels

        scores = self(given_panels, answer_panels, idx=dataloader_idx + self.increment_dataloader_idx)
        # print("scores and target>>",scores,target)
        pred = scores.argmax(1)
        # print(scores)
        # print(scores.shape) # batch x num_choices
        # print(f"Prediction: {pred}, Target: {target}")
        ce_loss = self.loss(scores, target) # cross entropy loss for slot image reconstruction

        current_metrics = self.additional_metrics[dataloader_idx + self.increment_dataloader_idx] # computing and reporting metrics
        for metric_nm, metric_func in current_metrics.items():
            value = metric_func(pred, target)
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/{metric_nm}",
                value,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
        if self.auxiliary_loss_ratio > 0:
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/mse_loss",
                slot_model_loss,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/cross_entropy_loss",
                ce_loss,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
        # acc = torch.eq(pred,target).float().mean().item() * 100.0

        # print("mse loss>>>",mse_criterion(torch.stack(recon_combined_seq,dim=1).squeeze(4), img))
        # print("ce loss>>",ce_criterion(scores,target))
        # print("recon combined seq shape>>",torch.stack(recon_combined_seq,dim=1).shape)
        # loss = 1000*mse_criterion(torch.stack(recon_combined_seq,dim=1), img) + ce_criterion(scores,target)
        # loss = ce_criterion(scores,target)
        loss = ce_loss + self.auxiliary_loss_ratio * slot_model_loss
        return loss

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
