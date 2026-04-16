from pathlib import Path
from dotenv import load_dotenv

import torch

torch.set_float32_matmul_precision("high")  # high / medium
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

from callbacks.motion_viz import MotionVizCallback
from callbacks.viz import TrajectoryVisualizationCallback


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=Path(cfg.ckpt_dir),
        filename="epoch_{epoch:02d}_loss_{val/loss:.4f}",
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

    motion_viz = MotionVizCallback(every_n_epochs=1)
    callbacks.append(motion_viz)

    return callbacks


@hydra.main(version_base=None, config_path="configs", config_name="config_vectornet")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    dm = instantiate(cfg.datamodule)
    model = instantiate(cfg.model, lr=cfg.optimizer.lr)

    profiler = None
    if cfg.trainer.profiler is not None:
        cfg.trainer.max_epochs = 3

    logger = instantiate(cfg.logger)
    logger.log_hyperparams(cfg)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=build_callbacks(cfg),
        **cfg.trainer,
    )

    mode = str(cfg.get("mode", "train")).lower()
    if mode == "train":
        trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("resume_from"))
    elif mode == "eval":
        ckpt_path = cfg.get("resume_from")
        if ckpt_path is None:
            raise ValueError(
                "In eval mode, please provide checkpoint via eval_ckpt=... or resume_from=..."
            )
        trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of: train, eval")


if __name__ == "__main__":
    load_dotenv()
    main()
