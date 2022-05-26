import numpy as np
import torch
from sklearn.metrics import roc_curve

from models.resnet import train_dropout


def get_baseline_outputs(model, config, data_module):
    test_output_baseline = gather_all_outputs(
        model=model,
        dataloader=data_module.test_dataloader(),
        n_classes=config.n_classes,
    )

    train_output_baseline = gather_all_outputs(
        model=model,
        dataloader=data_module.train_dataloader(),
        n_classes=config.n_classes,
    )

    if config.n_classes <= 2:
        val_out = gather_all_outputs(
            model=model,
            dataloader=data_module.val_dataloader(),
            n_classes=config.n_classes,
        )
        threshold = get_threshold(val_out["probas"], val_out["targets"], target_fpr=0.2)
        test_output_baseline["predictions"] = (
            test_output_baseline["probas"][:, 1] > threshold
        )
        train_output_baseline["predictions"] = (
            train_output_baseline["probas"][:, 1] > threshold
        )
    else:
        test_output_baseline["predictions"] = torch.argmax(
            test_output_baseline["probas"], 1
        )
        train_output_baseline["predictions"] = torch.argmax(
            train_output_baseline["probas"], 1
        )
        threshold = 0.5

    return test_output_baseline, train_output_baseline, threshold


def gather_single_pass_outputs(model, dataloader, n_classes):
    """
    Get all logits and features maps from a trained model on a given dataloader.
    Assumes the model has "get_feautures" and a "classify_features" methods (see classification/models)
    """
    all_targets, all_probas, all_feats = [], [], []

    for data, target in dataloader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            feats = model.get_features(data)
            logits = model.classify_features(feats)

        if n_classes > 1:
            probas = torch.softmax(logits, dim=1)
        else:
            probas = torch.stack(
                [1 - torch.sigmoid(logits), torch.sigmoid(logits)], dim=1
            )

        all_targets.append(target)
        all_probas.append(probas)
        all_feats.append(feats)

    probas = torch.cat(all_probas).cpu()
    targets = torch.cat(all_targets).cpu()
    feats = torch.cat(all_feats).cpu()

    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.reshape(-1)

    return {
        "probas": probas,
        "targets": targets,
        "feats": feats,
    }


def gather_all_outputs(
    model,
    dataloader,
    n_classes,
    n_inference_passes=1,
    mcmc_dropout: bool = False,
):
    """
    Runs multiple inference passes through the model with dropout at test time.
    Returns average prediction and average features, same format as output of gather_all_outputs.
    """
    dataloader.shuffle = False
    model.dropout_at_test_time = mcmc_dropout
    if mcmc_dropout:
        model.apply(train_dropout)
    test_outs = [
        gather_single_pass_outputs(
            model=model, dataloader=dataloader, n_classes=n_classes
        )
        for _ in range(n_inference_passes)
    ]
    model.dropout_at_test_time = False
    model.eval()  # To reset the dropout layers.
    list_probas = [t["probas"] for t in test_outs]
    mean_probas = torch.stack(list_probas, dim=0).mean(dim=0)
    return {
        "probas": mean_probas,
        "feats": [t["feats"] for t in test_outs]
        if len(test_outs) > 1
        else test_outs[0]["feats"],
        "targets": test_outs[0]["targets"],
    }


def gather_ensemble_outputs(models, dataloader, n_classes):
    """
    Runs multiple inference passes through the model with dropout at test time.
    Returns average prediction and average features, same format as output of gather_all_outputs.
    """
    dataloader.shuffle = False
    test_outs = []
    for model in models:
        assert not model.training, "Model should be in eval mode"
        if torch.cuda.is_available():
            model = model.cuda()
        test_outs.append(
            gather_all_outputs(model=model, dataloader=dataloader, n_classes=n_classes)
        )
        model = model.cpu()  # for memory management issues

    list_probas = [t["probas"] for t in test_outs]
    mean_probas = torch.stack(list_probas, dim=0).mean(dim=0)
    return {
        "probas": mean_probas,
        "feats": [t["feats"] for t in test_outs],
        "targets": test_outs[0]["targets"],
    }


def get_threshold(probas, targets, target_fpr):
    assert probas.shape[1] == 2, "Should be binary"
    if target_fpr is not None:
        fpr, _, thres = roc_curve(targets, probas[:, 1])
        return thres[np.argmin(np.abs(fpr - target_fpr))]
    return 0.5
