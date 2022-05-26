from copy import deepcopy
from pathlib import Path

import pandas as pd
import papermill as pm

import torch
from configs.load_model_and_config import (
    get_config_data_model_for_eval,
    get_failure_detection_filename,
)

from failure_detection.gather_results_utils import get_baseline_outputs
from failure_detection.uncertainty_scores.confidNet.scorer import get_confidnet_scores
from failure_detection.uncertainty_scores.laplace import get_laplace_scores
from failure_detection.uncertainty_scores.mcmc_scores import get_mcmc_scores
from failure_detection.uncertainty_scores.duq import get_duq_scores, load_duq_model
from failure_detection.uncertainty_scores.softmax_scorer import (
    get_baseline_scores,
    get_ensemble_scores,
    get_doctor_alpha_scores,
)
from failure_detection.uncertainty_scores.swag_scores import (
    get_swag_model_for_eval,
    get_swag_scores,
)
from failure_detection.uncertainty_scores.trust_score import get_trustscores


def evaluate_single_model(
    model, output_dir, config, config_name, data_module, execute_notebook
):
    print(torch.cuda.memory_allocated(0) / 104900.0)
    if torch.cuda.is_available():
        model.cuda()

    test_output_baseline, train_output_baseline, threshold = get_baseline_outputs(
        model, config, data_module
    )

    if get_failure_detection_filename(output_dir).exists():
        print("Found existing scores dict, updating only if necessary")
        scores_dataframe = pd.read_csv(get_failure_detection_filename(output_dir))
        scores_dict = scores_dataframe.to_dict(orient="list")
    else:
        scores_dict = get_baseline_scores(test_output_baseline, threshold)

    print(torch.cuda.memory_allocated(0) / 104900.0)

    if "doctor_alpha" in scores_dict.keys():
        print("Already have doctor alpha")
    else:
        scores_dict["doctor_alpha"] = get_doctor_alpha_scores(test_output_baseline)

    # If the model has dropout, get MCMC dropout scores
    if getattr(model, "p_dropout", 0) > 0:
        if "mcmc_soft_scores" in scores_dict.keys():
            print("MCMC dropout already calculated")
        else:
            assert hasattr(
                model, "dropout_at_test_time"
            ), "Model is missing dropout_at_test_time_attribute"
            print("Gathering MCMC dropout results")
            scores_dict.update(
                get_mcmc_scores(
                    model,
                    config.n_classes,
                    data_module.test_dataloader(),
                    data_module.val_dataloader(),
                )
            )

    # Get Laplace scores
    if "Laplace_predictions" in scores_dict.keys():
        print("Laplace has already been computed")
    else:
        print("Getting Laplace score")
        laplace_scores = get_laplace_scores(
            deepcopy(model),
            data_module.train_dataloader(),
            data_module.test_dataloader(),
            data_module.val_dataloader(),
            config.n_classes,
        )
        scores_dict.update(laplace_scores)
        print(torch.cuda.memory_allocated(0) / 104900.0)
    model = model.cpu()  # to avoid having several copies of the model on GPU.

    # Get TrustScore
    if "TrustScore" in scores_dict.keys():
        print("TrustScore has already been calculated")
    else:
        print("Gathering TrustScore results")
        trust_scores = get_trustscores(
            train_output_baseline["feats"],
            train_output_baseline["targets"],
            test_output_baseline["feats"],
            test_output_baseline["predictions"],
        )
        scores_dict.update(trust_scores)

    # Get ConfidNet scores
    if "ConfidNet_scores" in scores_dict.keys():
        print("Already gotten confidNet scores")
    else:
        print("Getting ConfidNet scores")
        confidnet_scores = get_confidnet_scores(
            deepcopy(model), data_module.test_dataloader(), output_dir
        )
        scores_dict.update(confidnet_scores)

    # Get SWAG
    if "SWAG_score" in scores_dict.keys():
        print("Already gotten SWAG scores")
    else:
        swag_model = get_swag_model_for_eval(
            deepcopy(model), output_dir, config.training.swag.num_epochs
        )
        if swag_model is not None:
            print("Getting SWAG scores")
            swag_scores = get_swag_scores(
                swag_model,
                data_module.train_dataloader(),
                data_module.test_dataloader(),
                data_module.val_dataloader(),
                config.n_classes,
            )
            scores_dict.update(swag_scores)

    # Get DUQ
    if "DUQ_score" in scores_dict.keys():
        print("Already have DUQ scores")
    else:
        duq_model = load_duq_model(deepcopy(model), output_dir, config)
        if duq_model is not None:
            print("Getting DUQ scores")
            duq_scores = get_duq_scores(
                duq_model,
                data_module.test_dataloader(),
            )
            scores_dict.update(duq_scores)

    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(0) / 104900.0)
    scores_dataframe = pd.DataFrame(scores_dict)
    scores_dataframe.to_csv(get_failure_detection_filename(output_dir), index=False)

    if execute_notebook:
        pm.execute_notebook(
            str(Path(__file__).parent / "evaluate_baselines.ipynb"),
            str(Path(output_dir) / "failure_detection" / "evaluate_baseline.ipynb"),
            parameters={
                "config_name": config_name,
                "scores_filename": str(get_failure_detection_filename(output_dir)),
            },
        )
    return Path(output_dir) / "failure_detection" / "all_metrics.csv"


def get_all_scores(
    config_name: str,
    execute_notebook: bool = True,
):
    config, data_module, models, output_dirs = get_config_data_model_for_eval(
        config_name
    )
    all_dfs = []

    for model, output_dir in zip(models, output_dirs):
        all_dfs.append(
            evaluate_single_model(
                model,
                output_dir,
                config,
                config_name,
                data_module,
                execute_notebook,
            )
        )

    if len(models) > 1:
        # Aggregate results for individual scores
        print(all_dfs)
        all_dfs = [pd.read_csv(df_name) for df_name in all_dfs]
        aggregated = (
            pd.concat(all_dfs, ignore_index=True)
            .groupby(by="Scorer")
            .aggregate(["mean", "std"])
        )
        (Path(output_dir).parent / "failure_detection").mkdir(
            parents=True, exist_ok=True
        )
        aggregated.to_csv(
            Path(output_dir).parent / "failure_detection" / "aggregated.csv",
            index=True,
        )

        # Get ensemble results
        if (
            Path(output_dir).parent / "failure_detection" / "ensemble_metrics.csv"
        ).exists():
            print("Already have ensemble scores, skipping")
        else:
            scores_dataframe, all_metrics = get_ensemble_scores(
                models, data_module, config.n_classes, Path(output_dir).parent
            )
            filename = (
                Path(output_dir).parent / "failure_detection" / "scores_df_ensemble.csv"
            )
            scores_dataframe.to_csv(filename, index=False)
            all_metrics.to_csv(
                Path(output_dir).parent / "failure_detection" / "ensemble_metrics.csv"
            )
            aggregated = all_metrics.groupby(by="Scorer").aggregate(["mean", "std"])
            aggregated.to_csv(
                Path(output_dir).parent
                / "failure_detection"
                / "aggregated_ensemble_metrics.csv"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Path to config file characterising trained model to evaluate",
    )
    args = parser.parse_args()
    get_all_scores(config_name=args.config, execute_notebook=True)
