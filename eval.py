from pathlib import Path

import torch

torch.set_float32_matmul_precision("high")  # high / medium

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from callbacks.viz import TrajectoryVisualizationCallback


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    callbacks = []

    viz = TrajectoryVisualizationCallback(every_n_epochs=1)
    callbacks.append(viz)

    return callbacks


@hydra.main(version_base=None, config_path="configs", config_name="config_vectornet")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    dm = instantiate(cfg.datamodule)
    model = instantiate(cfg.model, lr=cfg.optimizer.lr)

    logger = instantiate(cfg.logger)
    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=build_callbacks(cfg),
        **cfg.trainer,
    )

    if "resume_from" not in cfg or cfg["resume_from"] is None:
        raise ValueError("Please provide 'resume_from' path in config for evaluation.")
    ckpt_path = cfg["resume_from"]

    # 1) 先跑 validation
    val_metrics = trainer.validate(model, datamodule=dm, ckpt_path=ckpt_path)
    print("Validation done:", val_metrics)

    # 2) 再跑 test
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    print("Test done:", test_metrics)


if __name__ == "__main__":
    main()
