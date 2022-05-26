import torch

from classification.duq.duq_model import DUQModel
from pathlib import Path


def load_duq_model(feature_extractor, output_dir, config):
    path_checkpoint_folder = Path(output_dir) / "DUQ"
    best = list(path_checkpoint_folder.glob("best*.pt"))
    if len(best) == 1:
        duq_model = DUQModel(
            feature_extractor,
            config.n_classes,
            feature_extractor.num_features,
            config.training.duq.length_scale,
            config.training.duq.gamma,
        )
        duq_checkpoint = torch.load(str(best[0]))
        duq_model.load_state_dict(duq_checkpoint)
        if torch.cuda.is_available():
            duq_model = duq_model.cuda()
        duq_model = duq_model.eval()
        return duq_model
    return None


def gather_duq_outputs(model, dataloader):
    """
    Get all logits and features maps from a trained DUQ model on a given dataloader.
    """
    all_targets, all_probas, all_feats = [], [], []

    for data, target in dataloader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            feats = model.get_features(data)
            rbf_out = model.classify_features(feats)

        all_targets.append(target)
        all_probas.append(rbf_out)
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


def get_duq_scores(duq_model, test_dataloader):
    test_out = gather_duq_outputs(duq_model, test_dataloader)
    base_duq_score = torch.max(test_out["probas"], dim=1)[0]
    return {
        "DUQ_score": base_duq_score,
        "DUQ_predictions": torch.argmax(test_out["probas"], 1),
        "DUQ_probas": test_out["probas"][:, 1]
        if test_out["probas"].shape[1] == 2
        else None,
    }
