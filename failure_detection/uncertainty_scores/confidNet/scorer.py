from pathlib import Path

import torch
from failure_detection.uncertainty_scores.confidNet.confidNet import ConfidNetModule


def get_confidnet_scores(model, test_loader, model_output_dir):
    confidnet_output_dir = Path(model_output_dir) / "ConfidNet"
    checkpoint_path = confidnet_output_dir / "best_val.ckpt"
    if not checkpoint_path.exists():
        print("No ConfidNet model - ignore computation")
        return {}
    print(f"Loading ConfidNet checkpoint from {checkpoint_path}")
    confidnet_module = (
        ConfidNetModule(model)
        .load_from_checkpoint(checkpoint_path, main_model=model)
        .eval()
        .cuda()
    )
    all_scores = []
    with torch.no_grad():
        for batch in test_loader:
            all_scores.append(confidnet_module.predict_step(batch).cpu())
    return {"ConfidNet_scores": torch.cat(all_scores, dim=0).reshape(-1)}
