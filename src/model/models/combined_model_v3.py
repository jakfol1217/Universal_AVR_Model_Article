from itertools import permutations


import numpy as np
import pytorch_lightning as pl
import torch
import timm 
import torch.nn as nn
from torch import autocast
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .scoring_model_v1 import ScoringModel



class CombinedModel(ScoringModel):

    def __init__(
            self,
            cfg: DictConfig,
            context_norm: bool,
            slot_model: pl.LightningModule,
            #slot_model_v3: pl.LightningModule,
            relational_module: pl.LightningModule | None = None,
            relational_scoring_module_1: pl.LightningModule | None = None,
            relational_scoring_module_2: pl.LightningModule | None = None,
            real_idxes: list = [0],
            additional_metrics: dict = {},
            save_hyperparameters: bool = True,
            freeze_slot_model: bool = True,
            limit_to_groups: bool = False,
            auxiliary_loss_ratio: float = 0.0,
            in_dim: int = 1024,
            use_separate_aggregators: bool = False,
            **kwargs,
    ):
        
        super().__init__(cfg,
            context_norm=context_norm, 
            num_correct=1, 
            in_dim=in_dim, 
            slot_model=slot_model,
            transformer=None, 
            pos_emb=None, 
            additional_metrics=additional_metrics,
            save_hyperparameters=save_hyperparameters, 
            freeze_slot_model=freeze_slot_model,
            auxiliary_loss_ratio=auxiliary_loss_ratio)
        
        # loading relational scoring module
        self.relational_scoring_module_1 = relational_scoring_module_1

        self.relational_scoring_module_2 = relational_scoring_module_2


        # relational model for real-life images
        self.relational_module = relational_module

        self.real_idxes = real_idxes # indexes of datasets with real-life images for training

        self.limit_to_groups = limit_to_groups # whether to limit relational computations to groups (e.g. computing realtions for only 1st group in bongard problems)

        self.dataloader_idx = kwargs.get("dataloader_idx")

        task_metrics_idxs = [ # loading additional metrics for different tasks from configuration files
            int(_it.removeprefix("task_metric_"))
            for _it in kwargs.keys()
            if _it.startswith("task_metric_")
        ]

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

        self.task_metrics = nn.ModuleList(  # loading additional metrics for different tasks
            [
                create_module_dict(kwargs.get(f"task_metric_{_ix}"))
                for _ix in task_metrics_idxs
            ]
        )

        if len(self.task_metrics) > 0:
            self.additional_metrics = self.task_metrics


        self.separate_aggregators = nn.ParameterList([nn.Parameter(data=torch.rand(2, requires_grad=True)) 
                                                          for _ in range(2)])
        
        
        
        self.use_separate_aggregators = use_separate_aggregators
        print(self.use_separate_aggregators)


        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        # loading modules from checkpoints
        self.relational_module = self.load_module_from_checkpoint(
                        cfg.model.relational_module.ckpt_path, 
                        "relational_module", 
                        relational_module
                        )
        

        self.relational_scoring_module_1 = self.load_module_from_checkpoint(
                        cfg.model.relational_scoring_module_1.ckpt_path,
                        "relational_scoring_module_1",
                        relational_scoring_module_1
                        )
        
        self.relational_scoring_module_2 = self.load_module_from_checkpoint(
                        cfg.model.relational_scoring_module_2.ckpt_path,
                        "relational_scoring_module_2",
                        relational_scoring_module_2
                        )
        
        

        # whether to freeze modules
        self.freeze_module(self.relational_module, cfg.model.relational_module.freeze_module)

        if cfg.model.relational_scoring_module_1.freeze_module:
            if cfg.model.relational_scoring_module_1.transformer is None:
                self.partial_freeze_module_mlp(self.relational_scoring_module_1, layers=cfg.model.relational_scoring_module_1.layers_to_train)
            else:
                self.partial_freeze_module(self.relational_scoring_module_1, layers=cfg.model.relational_scoring_module_1.layers_to_train)

        if cfg.model.relational_scoring_module_2.freeze_module:
            if cfg.model.relational_scoring_module_2.transformer is None:
                self.partial_freeze_module_mlp(self.relational_scoring_module_2, layers=cfg.model.relational_scoring_module_2.layers_to_train)
            else:
                self.partial_freeze_module(self.relational_scoring_module_2, layers=cfg.model.relational_scoring_module_2.layers_to_train)

                
    def load_module_from_checkpoint(self, module_ckpt_path, module_name, module):
        if module_ckpt_path is not None: # if path is not empty, load module from checkpoint
            ckpt = torch.load(module_ckpt_path)
            state_dict = {
                key.split(".", 1)[1]: val
                for key, val in ckpt["state_dict"].items()
                if key.startswith(module_name)

            }
            del ckpt
            
            module.load_state_dict(state_dict)

        return module
        

    def freeze_module(self, module, freeze):
        if freeze:
            module.freeze()
        else:
            module.unfreeze()
    

    def partial_freeze_module(self, module, layers):
        for param in module.parameters():
            param.requires_grad = False
        for layer in layers:
            for name, param in module.named_parameters():
                layer_name = f"transformer.layers.{layer}"
                if name.startswith(layer_name):
                    param.requires_grad=True


    def partial_freeze_module_mlp(self, module, layers):
        for param in module.parameters():
            param.requires_grad = False
        for layer in layers:
            for name, param in module.named_parameters():
                layer_name = f"scoring_mlp.{layer}"
                if name.startswith(layer_name):
                    param.requires_grad=True


    def is_task_abstract(self, image): # todo: how to detect if task real or abstract? for now it's hard-coded
        return image not in self.real_idxes

    def forward(self, given_panels, answer_panels, isAbstract, idx):
        # computing scores from realtional matrices
        if self.use_separate_aggregators:
            rel_matrix = self.relational_module(given_panels, answer_panels, self.separate_aggregators[idx])
        else:
            rel_matrix = self.relational_module(given_panels, answer_panels)
            
        if not isAbstract:
            scores = self.relational_scoring_module_1(rel_matrix)
        else:

            scores = self.relational_scoring_module_2(rel_matrix)
            
        return scores

        
    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        img, target = batch

        if self.dataloader_idx is not None:
            dataloader_idx = self.dataloader_idx

        isAbstract = self.is_task_abstract(dataloader_idx) # pass img, for now its hard coded

        slot_model_loss = None

        context_panels_cnt = self.cfg.data.tasks[ # number of task context panels
            self.task_names[min(len(self.task_names)-1, dataloader_idx)]
        ].num_context_panels


        given_panels = img[:, :context_panels_cnt] # context panels
        answer_panels = img[:, context_panels_cnt:] # answer panels


        context_groups = self.cfg.data.tasks[ # context groups (e.g. using only 1 group from bongard instead of all context images)
                self.task_names[min(len(self.task_names)-1, dataloader_idx)]
            ].context_groups
        
        if self.limit_to_groups: # whether to limit relational computing to groups (e.g. computing realtions for only 1st group in bongard problems)
            scores = self(given_panels[:, context_groups[0], :], answer_panels, isAbstract=isAbstract, idx=dataloader_idx)
        else:
            scores = self(given_panels, answer_panels, isAbstract=isAbstract, idx=dataloader_idx)
        
        softmax = nn.Softmax(dim=1)
        scores = softmax(scores)

        pred = scores.argmax(1)

        ce_loss = self.loss(scores, target)  # cross entropy loss for slot image reconstruction
        current_metrics = self.additional_metrics[dataloader_idx] # computing and reporting metrics
        for metric_nm, metric_func in current_metrics.items():
            value = metric_func(pred, target)
            self.log(
                f"{step_name}/{self.task_names[min(len(self.task_names)-1, dataloader_idx)]}/{metric_nm}",
                value,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
        if self.auxiliary_loss_ratio > 0:
            self.log(
                f"{step_name}/{self.task_names[min(len(self.task_names)-1, dataloader_idx)]}/mse_loss",
                slot_model_loss,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{step_name}/{self.task_names[min(len(self.task_names)-1, dataloader_idx)]}/cross_entropy_loss",
                ce_loss,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                add_dataloader_idx=False,
            )

        loss = ce_loss
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
