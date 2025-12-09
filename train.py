import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def build_callbacks(cfg: DictConfig) -> list[pl.Callback]:
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="./outputs/checkpoints",
        filename="vectornet-{epoch:02d}-{val/loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    dm = instantiate(cfg.datamodule)
    model = instantiate(cfg.model, lr=cfg.optimizer.lr)

    logger = TensorBoardLogger(
        save_dir=cfg.trainer.get("default_root_dir", "./outputs"), name="vectornet"
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=build_callbacks(cfg),
        **cfg.trainer,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
