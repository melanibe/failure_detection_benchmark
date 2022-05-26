from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.collections import MetricCollection
from torch.optim.lr_scheduler import ReduceLROnPlateau

# A generic module for all the models
class ClassificationModule(pl.LightningModule):
    """
    A generic PL module for classification
    """

    def __init__(
        self,
        n_classes,
        lr: float = 1e-3,
        weight_decay=1e-4,
        p_dropout: float = 0.0,
        milestones: list = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.p_dropout = p_dropout
        self.milestones = milestones
        metrics = self.get_metrics()
        self.train_metrics = (
            metrics.clone(prefix="Train/") if metrics is not None else None
        )
        self.val_metrics = metrics.clone(prefix="Val/") if metrics is not None else None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = self.get_model()
        self.save_hyperparameters()

    def training_step(self, batch: Any, batch_idx: int) -> Any:  # type: ignore
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        self.log("Train/loss", loss, on_epoch=True, on_step=False)
        if self.train_metrics is not None:
            self.train_metrics.update(output.detach(), target)
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        if self.train_metrics is not None:
            self.train_metrics.compute()
            self.log_dict(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is not None:
            self.val_metrics.compute()
            self.log_dict(self.val_metrics)

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore
        data, target = batch
        output = self.model(data)
        loss = self.criterion(output, target)
        self.log("Val/loss", loss, on_epoch=True, on_step=False)
        if self.val_metrics is not None:
            self.val_metrics.update(output, target)

    def get_model(self) -> torch.nn.Module:
        raise NotImplementedError(
            "Get model should be implemented in the child classes"
        )

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        if self.milestones is not None:
            scheduler = MultiStepLR(optimizer, milestones=self.milestones)
            return [optimizer], [scheduler]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, patience=10),
                    "monitor": "Val/loss",
                },
            }
        return optimizer

    def get_metrics(self):
        metric_list = [torchmetrics.Accuracy()]
        metric_list.append(torchmetrics.AUROC(num_classes=self.n_classes))
        return MetricCollection(metric_list)
