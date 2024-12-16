import logging
import os
import sys

import config  # config register OmegaConf resolvers (DO NOT REMOVE IT)
import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from model.avr_datasets import IRAVENdataset
from model.models.STSN import SlotAttentionAutoEncoder
from omegaconf import DictConfig, OmegaConf
from wandb_agent import WandbAgent

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

def extract_wandb_id(id):
    paths = [
        # "/mnt/evafs/groups/mandziuk-lab/akaminski/logs",
        # "/home2/faculty/akaminski/Universal_AVR_Model/logs",
        # "/home2/faculty/jfoltyn/Universal_AVR_Model/logs",
        "/app/logs"
    ]
    for path in paths:
        try:
            wandb_id = WandbAgent.extract_wandb_id(id, log_dir=path)
            return wandb_id
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Could not find wandb_id for {id}")

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def _test(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.torch.matmul_precision)  # 'medium' | 'high'
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    pl.seed_everything(cfg.seed)
    data_module = instantiate(cfg.data.datamodule, cfg)
    torch.set_default_device("cuda")
    torch.multiprocessing .set_start_method('spawn')

    # TODO: checkpoint mechanism (param in config + loading from checkpoint)
    # TODO: datamodules (combination investiagtion)

    wandb_id=None
    if cfg.checkpoint_path is not None:
        _id = int(cfg.checkpoint_path.split("/")[-2])
        
        wandb_id = extract_wandb_id(_id)
    wandb_logger = WandbLogger(project="AVR_universal", log_model=False, id=wandb_id)

    # module = instantiate(cfg.model, cfg)
    module_kwargs = {}
    # print(cfg.model._target_)
    for k in cfg.model.keys():
        # print(k)
        if (
            isinstance(cfg.model[k], DictConfig)
            and cfg.model.get(k).get("_target_") is not None
        ):
            module_kwargs[k] = instantiate(cfg.model[k], cfg)
        else:
            module_kwargs[k] = cfg.model[k]
    print(cfg.model)
    module = instantiate(cfg.model, cfg, **module_kwargs, _recursive_=False)

    if cfg.checkpoint_path is not None:
        module = module.__class__.load_from_checkpoint(
            cfg.checkpoint_path, cfg=cfg, **module_kwargs, _recursive_=False
        )
    # print(module.device)

    print(module)
    # print(cfg.trainer)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.logger = wandb_logger
    wandb_logger.watch(module)

    trainer.fit(module, data_module, ckpt_path=cfg.checkpoint_path)
    # trainer.test(module, data_module)

    # example loading best model from newest run
    # wandb_agent = WandbAgent("AVR_universal")
    # checkpoint_path = wandb_agent.get_newest_checkpoint()
    # new_model = SlotAttentionAutoEncoder.load_from_checkpoint(checkpoint_path, cfg=cfg)
    # print(new_model)

    # The way to access ``hydra.**`` configuration
    # print(HydraConfig.get().launcher)
    # print(HydraConfig.get().launcher.partition)


if __name__ == "__main__":
    load_dotenv()
    _test()
