import itertools
import pickle
from typing import List
import numpy as np
import pandas as pd
import torch
from failure_detection.evaluator import ThresholdBasedEvaluator
from failure_detection.gather_results_utils import (
    gather_ensemble_outputs,
    get_threshold,
)


def get_baseline_scores(test_output_baseline, threshold):
    baseline_scorer = SoftmaxBasedScorer(threshold)
    scores_dict = {
        "Targets": test_output_baseline["targets"],
        "Predictions": test_output_baseline["predictions"],
        "IsCorrect": test_output_baseline["predictions"]
        == test_output_baseline["targets"],
    }

    if test_output_baseline["probas"].shape[1] == 2:
        scores_dict["Probas"] = test_output_baseline["probas"][:, 1]
        scores_dict["Threshold"] = threshold

    # Get baseline scores
    scores_dict["Baseline"] = baseline_scorer.get_scores(test_output_baseline["probas"])
    return scores_dict


def get_doctor_alpha_scores(test_output_baseline):
    g_x = torch.sum(test_output_baseline["probas"] ** 2, dim=1)
    return -(1 - g_x) / g_x


def get_doctor_beta_scores(softmax_scores):
    softmax_scores = torch.tensor(softmax_scores)
    return -(1 - softmax_scores) / softmax_scores


def get_ensemble_scores(list_models, data_module, n_classes, output_dir):
    all_combinations = list(itertools.combinations(list(range(len(list_models))), 3))
    sampled_combination = np.random.permutation(all_combinations)[:5]
    all_dfs, all_metrics_df = [], []
    for i, current_models_idx in enumerate(sampled_combination):
        current_list_models = np.asarray(list_models)[current_models_idx]
        ensemble_outputs = gather_ensemble_outputs(
            current_list_models, data_module.test_dataloader(), n_classes
        )
        if n_classes <= 2:
            val_ensemble_outputs = gather_ensemble_outputs(
                current_list_models, data_module.val_dataloader(), n_classes
            )
            ensemble_threshold = get_threshold(
                val_ensemble_outputs["probas"], val_ensemble_outputs["targets"], 0.2
            )
        else:
            ensemble_threshold = 0.5
        ensemble_outputs["predictions"] = (
            ensemble_outputs["probas"][:, 1] > ensemble_threshold
            if n_classes <= 2
            else torch.argmax(ensemble_outputs["probas"], 1)
        )
        with open(str(output_dir / f"ensemble_{i}.pickle"), "wb") as handle:
            pickle.dump(ensemble_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ensemble_scores = get_baseline_scores(ensemble_outputs, ensemble_threshold)
        current_df = pd.DataFrame()
        current_df["Targets"] = ensemble_outputs["targets"]
        current_df["ensemble_scores"] = ensemble_scores["Baseline"]
        current_df["ensemble_predictions"] = ensemble_outputs["predictions"]
        current_df["ensemble_threshold"] = ensemble_threshold
        current_df["ensemble_probas"] = (
            ensemble_outputs["probas"][:, 1] if n_classes <= 2 else None
        )
        current_df["N_ens"] = i
        all_dfs.append(current_df)
        evaluator = ThresholdBasedEvaluator(
            current_df["ensemble_scores"],
            current_df["ensemble_predictions"],
            current_df["Targets"],
            current_df["ensemble_probas"],
            "Ensemble",
        )

        metrics_df = evaluator.get_new_metrics()
        metrics_df["N_ens"] = i
        all_metrics_df.append(metrics_df)

    return pd.concat(all_dfs), pd.concat(all_metrics_df)


class SoftmaxBasedScorer:
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def get_scores(self, probabilities):
        if isinstance(probabilities, List):
            probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
        if probabilities.ndim == 1:
            print("Binary case")
            scores = probabilities.clone()
            scores[probabilities < self.threshold] = (
                1 - scores[probabilities < self.threshold]
            )
        else:
            scores = torch.max(probabilities, dim=1)[0]
        return scores


class EntropyScorer:
    def get_scores(self, probabilities):
        """
        Returns neg entropy of probabilty distirbution. If provided with a list of several
        inference passses predictions, first compute mean and then compute entropy of the mean.
        """
        if isinstance(probabilities, List):
            probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
        neg_entropy = (probabilities * torch.log(probabilities + 1e-9)).sum(dim=1)
        return neg_entropy
