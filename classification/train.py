from pathlib import Path

import pytorch_lightning as pl
import torch
from configs.default_config import load_yaml_training_config
from configs.load_model_and_config import get_modules, get_output_dir_from_config
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Path to config file characterising trained CNN model/s",
    )
    args = parser.parse_args()

    config = load_yaml_training_config(
        Path(__file__).parent.parent / "configs" / args.config
    )

    if isinstance(config.seed, int):
        config.seed = [config.seed]
    all_seeds = config.seed

    for seed in all_seeds:
        config.seed = seed
        output_dir = get_output_dir_from_config(config)

        pl.seed_everything(config.seed, workers=True)

        data_module, model_module = get_modules(config)

        checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename="{epoch}")
        checkpoint_callback_best = ModelCheckpoint(
            dirpath=output_dir, monitor="Val/loss", mode="min", filename="best"
        )
        lr_monitor = LearningRateMonitor()
        early_stopping = EarlyStopping(monitor="Val/loss", patience=15)
        logger = TensorBoardLogger(output_dir, name="tensorboard")
        n_gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            gpus=n_gpus,
            max_epochs=config.training.num_epochs,
            logger=logger,
            callbacks=[
                checkpoint_callback,
                checkpoint_callback_best,
                lr_monitor,
                early_stopping,
            ],
        )
        trainer.fit(model_module, data_module)
