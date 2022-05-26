from failure_detection.uncertainty_scores.softmax_scorer import (
    get_threshold,
    SoftmaxBasedScorer,
)
from laplace import Laplace
import torch


def get_laplace_scores(model, train_loader, test_loader, val_loader, n_classes):
    # Fit Laplace on training set
    la = Laplace(model, "classification")
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")

    # Get predictions on test dataloader
    test_probas, test_targets = [], []
    for img, target in test_loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
            result = la(img).cpu()  # .cuda()
            test_probas.append(result)  # laplace gives by probabilities
            test_targets.append(target)
    test_probas, test_targets = (
        torch.cat(test_probas).cpu(),
        torch.cat(test_targets).cpu(),
    )
    test_out = {"Laplace_targets": test_targets}

    # If binary, compute threshold on val set and get scores
    if n_classes <= 2:
        val_probas, val_targets = [], []
        for img, targets in val_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    img = img.cuda()
                result = la(img).cpu()
                val_probas.append(result)  # laplace gives by probabilities
                val_targets.append(targets)
        val_probas, val_targets = (
            torch.cat(val_probas).cpu(),
            torch.cat(val_targets).cpu(),
        )
        threshold = get_threshold(val_probas, val_targets, target_fpr=0.2)
        test_out["Laplace_predictions"] = test_probas[:, 1] > threshold
        test_out["Laplace_threshold"] = threshold
        test_out["Laplace_score"] = SoftmaxBasedScorer(threshold).get_scores(
            test_probas
        )
        test_out["Laplace_probas"] = test_probas[:, 1]
    # Else, get scores
    else:
        test_out["Laplace_predictions"] = torch.argmax(test_probas, 1)
        test_out["Laplace_score"] = SoftmaxBasedScorer().get_scores(test_probas)
    # Memory management on GPU seems to run into some issue with the laplace object
    # these 2 lines havex fixed it.
    la.model = la.model.cpu()
    del la
    return test_out
