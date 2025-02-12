import os
import re

import wandb


class WandbAgent:
    def __init__(self, project_name: str, **kwargs):
        self.api = wandb.Api(**kwargs)
        self.project_name = project_name

    def get_runs(self, **kwargs):
        return list(self.api.runs(self.project_name, **kwargs))

    def get_best_run_by_loss(self, key="val_loss", **kwargs):
        runs = self.get_runs(**kwargs)
        runs = [r for r in runs if key in r.summary]
        min_loss_run = min(runs, key=lambda run: run.summary[key])
        return min_loss_run

    def get_best_run_by_metric(self, key, **kwargs):
        runs = self.get_runs(**kwargs)
        runs = [r for r in runs if key in r.summary]
        max_metric_run = max(runs, key=lambda run: run.summary[key])
        return max_metric_run

    def get_newest_runs(self, **kwargs):
        runs = self.get_runs(**kwargs)
        runs = [r for r in runs if "_timestamp" in r.summary]
        newest_runs = sorted(
            runs, key=lambda run: run.summary["_timestamp"], reverse=True
        )
        return newest_runs

    def get_newest_checkpoint(self, **kwargs):
        runs = self.get_newest_runs(**kwargs)
        checkpoint = None
        for run in runs:
            try:
                checkpoint = self.get_best_checkpoint_from_run(run.name)
                if checkpoint:
                    break
            except ValueError:
                continue

        if checkpoint:
            return checkpoint
        else:
            raise ValueError(f"No run has viable model checkpoints")

    def get_run_by_name(self, run_name):
        runs = self.api.runs(self.project_name, filters={"display_name": run_name})
        if len(runs) == 1:
            return runs[0]
        else:
            raise ValueError(
                f"Query for runs with display_name='{run_name}' returned {len(runs)} results."
            )

    def get_run_by_id(self, run_id):
        run = self.api.run(f"{self.project_name}/{run_id}")
        return run

    def get_artifact_by_name(self, artifact_name):
        artifact = self.api.artifact(
            f"{self.project_name}/{artifact_name}", type="model"
        )
        return artifact

    def get_run_artifacts(self, run_name: str):
        run = self.get_run_by_name(run_name)
        artifacts = run.logged_artifacts()
        return artifacts

    def get_best_checkpoint_from_run(self, run_name):
        artifacts = self.get_run_artifacts(run_name)
        model_artifacts = [a for a in artifacts if a.type == "model"]
        if model_artifacts:
            for art in model_artifacts:
                if "best" in art.aliases:
                    return WandbAgent.download_checkpoint(art)
        else:
            raise ValueError(f"Run {run_name} has no model artifacts")
        # if best not found, return latest
        return WandbAgent.download_checkpoint(model_artifacts[-1])

    @staticmethod
    def download_checkpoint(artifact):
        checkpoint_path = artifact.download()
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, "model.ckpt")
        return checkpoint_path

    @staticmethod
    def extract_wandb_id(slurm_id: int, log_dir: str = "./logs") -> str:
        id = None
        with open(f"{log_dir}/slurm-{slurm_id}.log", "r") as file:
            line = file.readline()
            while line:
                match = re.search(r"View run at.*runs/(\w+)", line)
                if match:
                    id = match.group(1)
                    break
                line = file.readline()
        return id


if __name__ == "__main__":
    agent = WandbAgent("AVR_universal")
    # runs = agent.get_runs()
    # print(runs[0].summary)
    checkpoint_path = agent.get_newest_checkpoint()

    # print(run.summary)
# print(checkpoint_path)
