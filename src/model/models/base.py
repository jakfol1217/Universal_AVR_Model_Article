from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class AVRModule(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.task_names = list(self.cfg.data.tasks.keys())

        # self.save_hyperparameters(
        #     OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        # )

    @abstractmethod
    def _step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def module_step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        hydra_cfg = HydraConfig.get()
        module_name = OmegaConf.to_container(hydra_cfg.runtime.choices)[
            "data/datamodule"
        ]
        match module_name:
            case "multi_module":
                return self._multi_module_step(
                    step_name, batch, batch_idx, dataloader_idx
                )
            case "single_module":
                return self._single_module_step(
                    step_name, batch, batch_idx, dataloader_idx
                )
            case _:
                # TODO: additional steps for combined modules
                raise NotImplementedError(
                    f"Step function for {module_name} module not implemented yet"
                )

    def _single_module_step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            loss = self._step(
                step_name, batch[self.task_names[0]], batch_idx, dataloader_idx
            )
        else:
            loss = self._step(step_name, batch, batch_idx)
        self.log(
            f"{step_name}/{self.task_names[dataloader_idx]}/loss",
            loss.to(self.device),
            on_epoch=True,
            add_dataloader_idx=False,
            sync_dist=True
        )
        return loss

    def _multi_module_step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        loss = torch.tensor(0.0, device=self.device)

        if step_name == "train":
            for task_name in self.task_names:
                target_loss = self._step(
                    step_name,
                    batch[task_name],
                    batch_idx,
                    self.task_names.index(task_name),
                )
                loss += self.cfg.data.tasks[task_name].target_loss_ratio * target_loss
                self.log(
                    f"{step_name}/{task_name}/loss",
                    loss.to(self.device),
                    on_epoch=True,
                    add_dataloader_idx=False,
                    sync_dist=True
                )
        else:
            loss = self._step(step_name, batch, batch_idx, dataloader_idx)
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/loss",
                loss.to(self.device),
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True
            )
        return loss

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
