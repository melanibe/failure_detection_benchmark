"""
Code adapted from https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/train/run_swag.py
"""
import argparse
import time
from copy import deepcopy
from pathlib import Path

import tabulate
import torch
from classification.swag import losses, utils
from classification.swag.swag import SWAG
from configs.default_config import load_yaml_training_config
from configs.load_model_and_config import get_modules, get_output_dir_from_config
from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter


def schedule(epoch: int) -> float:
    t = float(epoch) / config.training.swag.swa_start
    lr_ratio = config.training.swag.swa_lr / config.training.swag.lr_init
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return config.training.swag.lr_init * factor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Path to config file characterising trained CNN model/s",
    )
    args = parser.parse_args()

    config = load_yaml_training_config(
        Path(__file__).parent.parent.parent / "configs" / args.config
    )
    use_cuda = torch.cuda.is_available()
    all_seeds = config.seed
    for seed in all_seeds:
        config.seed = seed
        seed_everything(config.seed)
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
        data_module, model_module = get_modules(config)
        data_module.prepare_data()
        data_module.setup()
        output_dir = get_output_dir_from_config(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=output_dir)
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        model = model_module.model.to(device)

        swag_model = SWAG(
            deepcopy(model),
            no_cov_mat=False,
            max_num_models=20,
        ).to(device)

        # use a slightly modified loss function that allows input of model
        criterion = losses.cross_entropy

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.swag.lr_init,
            momentum=config.training.swag.momentum,
            weight_decay=config.training.weight_decay,
        )

        start_epoch = 0

        columns = [
            "ep",
            "lr",
            "tr_loss",
            "tr_acc",
            "te_loss",
            "te_acc",
            "time",
            "mem_usage",
        ]
        columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
        swag_res = {"loss": None, "accuracy": None}

        for epoch in range(start_epoch, config.training.swag.num_epochs):
            time_ep = time.time()
            lr = schedule(epoch)
            utils.adjust_learning_rate(optimizer, lr)
            train_res = utils.train_epoch(
                data_module.train_dataloader(),
                model,
                criterion,
                optimizer,
                cuda=use_cuda,
            )

            val_res = utils.eval(
                data_module.val_dataloader(), model, criterion, cuda=use_cuda
            )

            if (epoch + 1) > config.training.swag.swa_start:
                swag_model.collect_model(model)
                swag_model.sample(0.0)
                utils.bn_update(data_module.train_dataloader(), swag_model)
                swag_res = utils.eval(
                    data_module.val_dataloader(), swag_model, criterion
                )
                writer.add_scalar("swa/val_accuracy", swag_res["accuracy"], epoch)
                writer.add_scalar("swa/val_loss", swag_res["loss"], epoch)

            if (epoch + 1) % 50 == 0:
                utils.save_checkpoint(
                    output_dir,
                    epoch + 1,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
                utils.save_checkpoint(
                    output_dir,
                    epoch + 1,
                    name="swag",
                    state_dict=swag_model.state_dict(),
                )

            time_ep = time.time() - time_ep

            if use_cuda:
                memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)
            writer.add_scalar("train/loss", train_res["loss"], epoch)
            writer.add_scalar("val/loss", val_res["loss"], epoch)
            writer.add_scalar("train/accuracy", train_res["accuracy"], epoch)
            writer.add_scalar("val/accuracy", val_res["accuracy"], epoch)
            writer.add_scalar("lr", lr, epoch)
            values = [
                epoch + 1,
                lr,
                train_res["loss"],
                train_res["accuracy"],
                val_res["loss"],
                val_res["accuracy"],
                time_ep,
                memory_usage,
            ]
            values = (
                values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
            )
            table = tabulate.tabulate(
                [values], columns, tablefmt="simple", floatfmt="8.4f"
            )
            if epoch % 20 == 0:
                table = table.split("\n")
                table = "\n".join([table[1]] + table)
            else:
                table = table.split("\n")[2]
            print(table)

        if epoch == config.training.swag.num_epochs - 1:
            utils.save_checkpoint(
                output_dir,
                config.training.swag.num_epochs,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            if config.training.swag.num_epochs > config.training.swag.swa_start:
                utils.save_checkpoint(
                    output_dir,
                    config.training.swag.num_epochs,
                    name="swag",
                    state_dict=swag_model.state_dict(),
                )
