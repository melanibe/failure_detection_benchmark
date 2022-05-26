from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import AveragePrecision

from models.resnet import train_dropout


class ConfidNet(torch.nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.uncertainty1 = torch.nn.Linear(feature_size, 400)
        self.uncertainty2 = torch.nn.Linear(400, 400)
        self.uncertainty3 = torch.nn.Linear(400, 400)
        self.uncertainty4 = torch.nn.Linear(400, 400)
        self.uncertainty5 = torch.nn.Linear(400, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.uncertainty1(x))
        x = F.relu(self.uncertainty2(x))
        x = F.relu(self.uncertainty3(x))
        x = F.relu(self.uncertainty4(x))
        return self.uncertainty5(x)


class ConfidNetModule(pl.LightningModule):
    def __init__(self, main_model: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.confid_net = ConfidNet(main_model.num_features)
        self.main_model = main_model.eval()
        assert hasattr(self.main_model, "get_features")
        self.criterion = torch.nn.MSELoss()
        # self.save_hyperparameters()
        self.train_aupr = AveragePrecision()
        self.val_aupr = AveragePrecision()
        for param in self.main_model.parameters():
            param.requires_grad = False

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore
        data, true_class = batch
        with torch.no_grad():
            self.main_model.eval()
            self.main_model.apply(train_dropout)
            feats = self.main_model.get_features(data)
            probas = torch.softmax(self.main_model.classify_features(feats), dim=1)
            targets = probas[torch.arange(true_class.shape[0]), true_class.flatten()]
        output = torch.sigmoid(self.confid_net(feats).squeeze())
        loss = self.criterion(output, targets)
        self.log("Train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:  # type: ignore
        data, true_class = batch
        with torch.no_grad():
            self.main_model.eval()
            feats = self.main_model.get_features(data)
            probas = torch.softmax(self.main_model.classify_features(feats), dim=1)
            targets = probas[torch.arange(true_class.shape[0]), true_class.flatten()]
        output = torch.sigmoid(self.confid_net(feats).squeeze())
        loss = self.criterion(output, targets)
        predictions = torch.argmax(probas, dim=1)
        self.val_aupr(-output, predictions != true_class)
        self.log("Val/ErrorAUPR", self.val_aupr)
        self.log("Val/loss", loss, on_epoch=True, on_step=False)

    def predict_step(self, batch: Any) -> Any:
        data, _ = batch
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            self.main_model.eval()
            self.main_model.dropout_at_test_time = False  # type: ignore
            feats = self.main_model.get_features(data)
            output = self.confid_net(feats).squeeze()
        return torch.sigmoid(output)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.confid_net.parameters(), lr=1e-4)
        return optimizer
