from enum import Enum

import pytorch_lightning as pl
from hydra.utils import instantiate
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class SingleModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert len(cfg.data.tasks.keys()) == 1, "SingleModule only supports single task"
        self.cfg = cfg
        self.task_name = next(iter(cfg.data.tasks.keys()))
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = instantiate(
                self.cfg.data.tasks[self.task_name].dataset.train
            )
            self.val_dataset = instantiate(
                self.cfg.data.tasks[self.task_name].dataset.val
            )
        elif stage == "test":
            self.test_dataset = instantiate(
                self.cfg.data.tasks[self.task_name].dataset.test
            )

    def train_dataloader(self) -> dict[str, DataLoader]:
        return {
            self.task_name: instantiate(
                self.cfg.data.tasks[self.task_name].dataloader.train,
                dataset=self.train_dataset,
            )
        }

    def val_dataloader(self) -> dict[str, DataLoader]:
        return {
            self.task_name: instantiate(
                self.cfg.data.tasks[self.task_name].dataloader.val,
                dataset=self.val_dataset,
            )
        }

    def test_dataloader(self) -> dict[str, DataLoader]:
        return {
            self.task_name: instantiate(
                self.cfg.data.tasks[self.task_name].dataloader.test,
                dataset=self.test_dataset,
            )
        }


class MultiModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}
        self.task_names = list(cfg.keys())

    def setup(self, stage: str):
        if stage == "fit":
            for task_name, task_cfg in self.cfg.data.tasks.items():
                self.train_datasets[task_name] = instantiate(task_cfg.dataset.train)
                self.val_datasets[task_name] = instantiate(task_cfg.dataset.val)
        elif stage == "test":
            for task_name, task_cfg in self.cfg.data.tasks.items():
                self.test_datasets[task_name] = instantiate(task_cfg.dataset.test)
        return

    def train_dataloader(self) -> dict[str, DataLoader]:
        return {
            task_name: instantiate(
                self.cfg.data.tasks[task_name].dataloader.train,
                dataset=dataset,
            )
            for task_name, dataset in self.train_datasets.items()
        }

    def val_dataloader(self) -> dict[str, DataLoader]:
        return {
            task_name: instantiate(
                self.cfg.data.tasks[task_name].dataloader.val,
                dataset=dataset,
            )
            for task_name, dataset in self.val_datasets.items()
        }

    def test_dataloader(self) -> dict[str, DataLoader]:
        return {
            task_name: instantiate(
                self.cfg.data.tasks[task_name].dataloader.test,
                dataset=dataset,
            )
            for task_name, dataset in self.test_datasets.items()
        }


class CombinedModuleStrategy(Enum):
    MIN_SIZE = "min_size"
    MAX_SIZE_CYCLE = "max_size_cycle"
    MAX_SIZE = "max_size"
    SEQUENTIAL = "sequential"


class CombinedModule(pl.LightningDataModule):
    def __init__(self, strategy: CombinedModuleStrategy, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}
        self.task_names = list(cfg.keys())
        self.strategy = strategy

    def setup(self, stage: str):
        if stage == "fit":
            for task_name, task_cfg in self.cfg.data.tasks.items():
                self.train_datasets[task_name] = instantiate(task_cfg.dataset.train)
                self.val_datasets[task_name] = instantiate(task_cfg.dataset.val)
        elif stage == "test":
            for task_name, task_cfg in self.cfg.data.tasks.items():
                self.test_datasets[task_name] = instantiate(task_cfg.dataset.test)
        return

    def train_dataloader(self):
        dataloaders = {
            task_name: instantiate(
                self.cfg.data.tasks[task_name].dataloader.train,
                dataset=dataset,
            )
            for task_name, dataset in self.train_datasets.items()
        }

        return CombinedLoader(dataloaders, self.strategy.value)

    def val_dataloader(self):
        dataloaders = {
            task_name: instantiate(
                self.cfg.data.tasks[task_name].dataloader.val,
                dataset=dataset,
            )
            for task_name, dataset in self.val_datasets.items()
        }

        return CombinedLoader(dataloaders, self.strategy.value)

    def test_dataloader(self):
        dataloaders = {
            task_name: instantiate(
                self.cfg.data.tasks[task_name].dataloader.test,
                dataset=dataset,
            )
            for task_name, dataset in self.test_datasets.items()
        }

        return CombinedLoader(dataloaders, self.strategy.value)


class CombinedModuleMinSize(CombinedModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(CombinedModuleStrategy.MIN_SIZE, cfg)


class CombinedModuleMaxSizeCycle(CombinedModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(CombinedModuleStrategy.MAX_SIZE_CYCLE, cfg)


class CombinedModuleMaxSize(CombinedModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(CombinedModuleStrategy.MAX_SIZE, cfg)


class CombinedModuleSequential(CombinedModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(CombinedModuleStrategy.SEQUENTIAL, cfg)
