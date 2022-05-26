from sklearn.metrics import det_curve, roc_auc_score
import numpy as np
import torch
import pandas as pd


class ThresholdBasedEvaluator:
    # Warning: "predicted score" assumed to be for "the prediction is correct"
    def __init__(
        self, scores, predictions, targets, probabilities, scorer_name: str = ""
    ) -> None:
        self.scores = scores.numpy() if isinstance(scores, torch.Tensor) else scores
        self.predictions = (
            predictions.numpy()
            if isinstance(predictions, torch.Tensor)
            else predictions
        )
        self.targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets
        self.correctly_classified = self.predictions == self.targets
        self.scorer_name = scorer_name
        self.probabilities = probabilities
        self.n_classes = np.unique(self.targets).size

    def get_AUC(self):
        return roc_auc_score(self.correctly_classified, self.scores)

    def get_FPR_at_TPR(self, target_tpr: float = 0.95):
        fpr, fnr, _ = det_curve(self.correctly_classified, self.scores)
        tpr = 1 - fnr
        index = np.argmin(tpr[tpr >= target_tpr])
        return fpr[index], tpr[index]

    def get_new_metrics(self) -> pd.DataFrame:
        fpr95, _ = self.get_FPR_at_TPR(0.95)
        fpr80, _ = self.get_FPR_at_TPR(0.80)
        metrics = pd.DataFrame(
            {
                "Scorer": self.scorer_name,
                "FPR_TPR95": fpr95,
                "FPR_TPR80": fpr80,
                "ROC_AUC_error_detection_global": self.get_AUC(),
                "Original_Accuracy": self.correctly_classified.mean(),
            },
            index=[0],
        )

        if self.n_classes == 2:
            metrics["Original_ROC_AUC"] = my_roc_score(self.targets, self.probabilities)
        return metrics


def my_roc_score(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan
