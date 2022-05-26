import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from configs.default_config import load_yaml_training_config
from configs.load_model_and_config import (
    get_modules,
    get_output_dir_from_config,
    load_model_from_checkpoint,
)
from failure_detection.uncertainty_scores.confidNet.confidNet import ConfidNetModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Path to config file characterising trained CNN model/s",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent.parent
    config = load_yaml_training_config(root / "configs" / args.config)

    all_seeds = config.seed

    for seed in all_seeds:
        config.seed = seed
        model_output_dir = get_output_dir_from_config(config)
        config.output_dir = str(model_output_dir)

        # Load classification model
        pl.seed_everything(config.seed, workers=True)
        data_module, model_module = get_modules(config)
        model = load_model_from_checkpoint(config, model_module)

        # Create ConfidNet module
        confidnet_module = ConfidNetModule(model)
        confidnet_output = model_output_dir / "ConfidNet"

        # Create trainer
        checkpoint_callback = ModelCheckpoint(
            dirpath=confidnet_output,
            filename="best_val",
            monitor="Val/ErrorAUPR",
            mode="max",
            save_last=True,
        )
        logger = TensorBoardLogger(str(confidnet_output), name="tensorboard")
        n_gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(
            gpus=n_gpus,
            max_epochs=250,
            logger=logger,
            callbacks=checkpoint_callback,
        )

        # Train
        trainer.fit(confidnet_module, data_module)
