from pathlib import Path

import torch
from configs.default_config import load_yaml_training_config
from configs.load_model_and_config import get_modules
from failure_detection.gather_results_utils import gather_single_pass_outputs
from models.resnet import train_dropout


def test_mcmc_dropout_inference():
    config = load_yaml_training_config(
        Path(__file__).parent.parent
        / "configs"
        / "medmnist"
        / "organamnist_resnet_dropout_all_layers.yml"
    )
    config.seed = config.seed[0]
    data_module, model_module = get_modules(config)
    model = model_module.model.eval()
    if torch.cuda.is_available():
        model.cuda()
    data_module.prepare_data()
    data_module.setup()
    data_loader = data_module.test_dataloader()

    # No MCMC-dropout should be no variability across inference passes
    test_outs = [
        gather_single_pass_outputs(
            model=model, dataloader=data_loader, n_classes=config.n_classes
        )
        for _ in range(3)
    ]
    list_probas = [t["probas"] for t in test_outs]
    assert torch.stack(list_probas, dim=0).std(dim=0).sum() == 0

    # Let's test MCMC dropout
    model.apply(train_dropout)
    test_outs = [
        gather_single_pass_outputs(
            model=model, dataloader=data_loader, n_classes=config.n_classes
        )
        for _ in range(3)
    ]
    list_probas = [t["probas"] for t in test_outs]
    assert torch.stack(list_probas, dim=0).std(dim=0).sum() != 0
