import torch
from failure_detection.gather_results_utils import gather_all_outputs, get_threshold
from failure_detection.uncertainty_scores.softmax_scorer import (
    EntropyScorer,
    SoftmaxBasedScorer,
)


def get_mcmc_scores(model, n_classes, test_loader, val_loader):
    mcmc_outputs_test = gather_all_outputs(
        model=model,
        dataloader=test_loader,
        n_classes=n_classes,
        n_inference_passes=10,
        mcmc_dropout=True,
    )

    # If binary, compute threshold on val set and get scores
    if n_classes <= 2:
        mcmc_outputs_val = gather_all_outputs(
            model=model,
            dataloader=val_loader,
            n_classes=n_classes,
            n_inference_passes=10,
            mcmc_dropout=True,
        )
        threshold = get_threshold(
            mcmc_outputs_val["probas"], mcmc_outputs_val["targets"], target_fpr=0.2
        )
        mcmc_outputs_test["predictions"] = mcmc_outputs_test["probas"][:, 1] > threshold
        mcmc_outputs_test["mcmc_threshold"] = threshold
    else:
        mcmc_outputs_test["predictions"] = torch.argmax(mcmc_outputs_test["probas"], 1)
        threshold = None
    scores_dict = {}
    scores_dict["mcmc_soft_scores"] = SoftmaxBasedScorer(threshold).get_scores(
        mcmc_outputs_test["probas"]
    )
    scores_dict["mcmc_predictions"] = mcmc_outputs_test["predictions"]
    scores_dict["mcmc_probas"] = (
        mcmc_outputs_test["probas"][:, 1]
        if mcmc_outputs_test["probas"].shape[1] == 2
        else None
    )
    scores_dict["mcmc_entropy_scores"] = EntropyScorer().get_scores(
        mcmc_outputs_test["probas"]
    )
    return scores_dict
