import torch
from failure_detection.uncertainty_scores.softmax_scorer import (
    SoftmaxBasedScorer,
    get_threshold,
)

from classification.swag import utils, swag
from pathlib import Path


def get_swag_model_for_eval(model, model_output_dir, num_epochs):
    if torch.cuda.is_available():
        model = model.cuda()
    path_checkpoint = Path(model_output_dir) / f"swag-{num_epochs}.pt"
    if path_checkpoint.exists():
        swag_checkpoint = torch.load(str(path_checkpoint))
        swag_model = swag.SWAG(model, no_cov_mat=False, max_num_models=20)
        swag_model.load_state_dict(swag_checkpoint["state_dict"])
        if torch.cuda.is_available():
            swag_model = swag_model.cuda()
        return swag_model
    return None


def gather_swag_probabilities(
    swag_model, test_loader, train_loader, n_inference, scale
):
    test_out = []
    for _ in range(n_inference):
        swag_model.sample(scale=scale, cov=True)
        utils.bn_update(train_loader, swag_model)
        swag_model.eval()
        test_probas, test_targets = [], []
        for img, target in test_loader:
            with torch.no_grad():
                if torch.cuda.is_available():
                    img = img.cuda()
                logits = swag_model(img).detach().cpu()
                test_probas.append(torch.softmax(logits, 1))
                test_targets.append(target)
        test_probas, test_targets = torch.cat(test_probas), torch.cat(test_targets)
        test_out.append({"SWAG_probas": test_probas, "SWAG_targets": test_targets})
    result = {
        "SWAG_probas": torch.stack([t["SWAG_probas"] for t in test_out], 0).mean(0),
        "SWAG_targets": test_out[0]["SWAG_targets"],
    }
    return result


def get_swag_scores(
    swag_model,
    train_loader,
    test_loader,
    val_loader,
    n_classes,
    n_inference=10,
    scale=0.5,
):
    # Get predictions on test dataloader
    test_out = gather_swag_probabilities(
        swag_model, test_loader, train_loader, n_inference=n_inference, scale=scale
    )
    # If binary, compute threshold on val set and get scores
    if n_classes <= 2:
        val_out = gather_swag_probabilities(
            swag_model, val_loader, train_loader, n_inference=n_inference, scale=scale
        )
        threshold = get_threshold(
            val_out["SWAG_probas"], val_out["SWAG_targets"], target_fpr=0.2
        )
        test_out["SWAG_predictions"] = test_out["SWAG_probas"][:, 1] > threshold
        test_out["SWAG_threshold"] = threshold
        test_out["SWAG_score"] = SoftmaxBasedScorer(threshold).get_scores(
            test_out["SWAG_probas"]
        )
        test_out["SWAG_probas"] = test_out["SWAG_probas"][:, 1]
    # Else, get scores
    else:
        test_out["SWAG_predictions"] = torch.argmax(test_out["SWAG_probas"], 1)
        test_out["SWAG_score"] = SoftmaxBasedScorer().get_scores(
            test_out["SWAG_probas"]
        )
        test_out.pop("SWAG_probas")
    swag_model = swag_model.cpu()
    return test_out
