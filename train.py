from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")  # high / medium
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

from callbacks.viz import TrajectoryVisualizationCallback


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=Path(cfg.output_root_dir) / "ckpt",
        filename="motion_prediction-{epoch:02d}-{val/loss:.4f}",
        save_top_k=3,
        save_last=True,
        mode="min",
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    viz = TrajectoryVisualizationCallback(every_n_epochs=1)
    callbacks.append(viz)

    return callbacks


@hydra.main(version_base=None, config_path="configs", config_name="config_vectornet")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    dm = instantiate(cfg.datamodule)
    model = instantiate(cfg.model, lr=cfg.optimizer.lr)

    profiler = None
    if cfg.trainer.profiler is not None:
        cfg.trainer.max_epochs = 3

    if cfg.logger.type == "wandb":
        logger = WandbLogger(
            project=cfg.project_name,
            name=cfg.exp_name,
            save_dir=cfg.output_root_dir,
            log_model=False,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=cfg.output_root_dir,
            name=f"{cfg.project_name}/{cfg.exp_name}",
        )

    logger.log_hyperparams(cfg)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=build_callbacks(cfg),
        **cfg.trainer,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("resume_from"))
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
