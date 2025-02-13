import logging
import os
import sys
from itertools import cycle

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import config  # config register OmegaConf resolvers (DO NOT REMOVE IT)
import wandb
from wandb_agent import WandbAgent

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


def setup_agent(key):
    return WandbAgent(project_name="AVR_universal", api_key=key)


def extract_wandb_id(id):
    paths = [
        "/app/logs", # "/mnt/evafs/groups/mandziuk-lab/akaminski/logs",
        "/home2/faculty/akaminski/Universal_AVR_Model/logs",
        "/home2/faculty/jfoltyn/Universal_AVR_Model/logs",
    ]
    for path in paths:
        try:
            wandb_id = WandbAgent.extract_wandb_id(id, log_dir=path)
            return wandb_id
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Could not find wandb_id for {id}")


@hydra.main(version_base=None, config_path="./conf", config_name="config_simple")
def main(cfg: DictConfig) -> None:
    assert cfg.checkpoint_path is not None
    torch.set_float32_matmul_precision(cfg.torch.matmul_precision)  # 'medium' | 'high'
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    keys = [os.environ["JF_WANDB_API_KEY"], os.environ["AK_WANDB_API_KEY"]]
    keys_cycle = cycle(keys)

    pl.seed_everything(cfg.seed)

    _id = int(cfg.checkpoint_path.split("/")[-2])
    wandb_id = extract_wandb_id(_id)

    curr_key = next(keys_cycle)
    wandb_agent = setup_agent(curr_key)
    run = None
    for _ in range(len(keys)):
        try:
            run = wandb_agent.get_run_by_id(run_id=wandb_id)
            break
        except Exception:
            curr_key = next(keys_cycle)
            wandb_agent = setup_agent(curr_key)
    if run is None:
        raise ValueError(f"Run {wandb_id} not found in any of the accounts")

    # TODO: maybe necessary to add this account switching to main.py
    os.environ["WANDB_API_KEY"] = curr_key
    wandb.login(key=curr_key, relogin=True)

    wandb_logger = WandbLogger(project="AVR_universal", log_model=False, id=wandb_id)

    # joining previous config with the new one (to avoid specifying technical details about model when we are only testing it)
    run_config = run.config
    run_config.update(cfg)
    run_config = DictConfig(run_config)
    data_module = instantiate(run_config.data.datamodule, run_config)

    module_kwargs = {}
    for k in run_config.model.keys():
        if (
            isinstance(run_config.model[k], (dict, DictConfig))
            and run_config.model.get(k).get("_target_") is not None
        ):
            print(run_config.model[k])
            module_kwargs[k] = instantiate(
                run_config.model[k], cfg=run_config, _recursive_=False
            )  # cfg/run_config, **cfg
        else:
            module_kwargs[k] = run_config.model[k]

    # not ideal solution, but it works
    additional_kwargs = {}
    if (increment_dataloader_idx := cfg.get("increment_dataloader_idx")):
        additional_kwargs["increment_dataloader_idx"] = increment_dataloader_idx

    print("cfg")
    print(cfg)
    
    print("run config")
    print(run_config)
    print("module kwargs")
    print(module_kwargs)
    print("additional kwargs")
    print(additional_kwargs)



    module_class = hydra.utils.get_class(run_config.model._target_)

    module = module_class.load_from_checkpoint(
        cfg.checkpoint_path, cfg=run_config, **module_kwargs, **additional_kwargs, _recursive_=False,
        dataloader_idx=cfg.get("dataloader_idx"), new_real_idxes=cfg.get("new_real_idxes")
    )

    print(module)

    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.logger = wandb_logger
    wandb_logger.watch(module)
    trainer.test(module, data_module)  # ckpt_path=cfg.checkpoint_path


if __name__ == "__main__":
    load_dotenv()
    main()
