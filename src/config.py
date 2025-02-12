from omegaconf import OmegaConf


def calculate_partition_type(timeout_min: int):
    if timeout_min < 60*24:
        return "short"
    else:
        return "long"

def get_slurm_job_id() -> str:
    import os
    
    if (job_id := os.getenv("SLURM_JOB_ID")) is not None:
        return job_id

    from datetime import datetime
    return f"{datetime.now():%Y%m%d_%H%M%S}"

OmegaConf.register_new_resolver("calculate_partition_type", calculate_partition_type)
OmegaConf.register_new_resolver("get_slurm_job_id", get_slurm_job_id)
