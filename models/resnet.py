import torch

from models import resnet_modified
from models.pl_module import ClassificationModule


def train_dropout(m):
    if type(m) == torch.nn.modules.dropout.Dropout:
        m.train()


class ResNet18(torch.nn.Module):
    def __init__(self, n_classes: int, p_dropout: float) -> None:
        super().__init__()
        self.net = resnet_modified.resnet18(num_classes=n_classes, p_dropout=p_dropout)
        self.num_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(self.num_features, n_classes)
        self.n_classes = n_classes
        self.p_dropout = p_dropout

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def classify_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(x)
        return self.classify_features(feats)


class ResNet50(ResNet18):
    def __init__(self, n_classes: int, p_dropout: float) -> None:
        super().__init__(n_classes, p_dropout)
        self.net = resnet_modified.resnet50(num_classes=n_classes, p_dropout=p_dropout)
        self.num_features = self.net.fc.in_features


class WideResnet50(ResNet18):
    def __init__(self, n_classes: int, p_dropout: float) -> None:
        super().__init__(n_classes, p_dropout)
        self.net = resnet_modified.wideresnet_50(
            num_classes=n_classes, p_dropout=p_dropout
        )
        self.num_features = self.net.fc.in_features


class ResNet18Module(ClassificationModule):
    def get_model(self) -> torch.nn.Module:
        return ResNet18(
            n_classes=self.n_classes,
            p_dropout=self.p_dropout,
        )


class ResNet50Module(ClassificationModule):
    def get_model(self) -> torch.nn.Module:
        return ResNet50(
            n_classes=self.n_classes,
            p_dropout=self.p_dropout,
        )


class WideResNet50Module(ClassificationModule):
    def get_model(self) -> torch.nn.Module:
        return WideResnet50(
            n_classes=self.n_classes,
            p_dropout=self.p_dropout,
        )
