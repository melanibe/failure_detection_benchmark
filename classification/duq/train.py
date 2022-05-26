"""
Adapted from https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/train_duq_cifar.py
"""

import argparse
import pathlib

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from classification.duq.duq_model import DUQModel

from configs.default_config import load_yaml_training_config
from configs.load_model_and_config import get_modules, get_output_dir_from_config
from pytorch_lightning import seed_everything
from ignite.handlers import Checkpoint, global_step_from_engine


def main(config):
    config = load_yaml_training_config(
        pathlib.Path(__file__).parent.parent.parent / "configs" / args.config
    )
    all_seeds = config.seed
    use_cuda = torch.cuda.is_available()
    for seed in all_seeds:

        config.seed = seed
        seed_everything(config.seed, workers=True)
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
        data_module, model_module = get_modules(config)
        data_module.prepare_data()
        data_module.setup()
        output_dir = get_output_dir_from_config(config) / "DUQ"
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=output_dir)
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        feature_extractor = model_module.model.to(device)
        model_output_size = feature_extractor.num_features

        model = DUQModel(
            feature_extractor,
            config.n_classes,
            model_output_size,
            config.training.duq.length_scale,
            config.training.duq.gamma,
        )

        if use_cuda:
            model = model.cuda()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.duq.lr,
            momentum=0.9,
            weight_decay=config.training.duq.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.training.duq.milestones, gamma=0.1
        )

        def calc_gradients_input(x, y_pred):
            gradients = torch.autograd.grad(
                outputs=y_pred,
                inputs=x,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True,
            )[0]

            gradients = gradients.flatten(start_dim=1)

            return gradients

        def calc_gradient_penalty(x, y_pred):
            gradients = calc_gradients_input(x, y_pred)

            # L2 norm
            grad_norm = gradients.norm(2, dim=1)

            # Two sided penalty
            gradient_penalty = ((grad_norm - 1) ** 2).mean()

            return gradient_penalty

        def step(engine, batch):
            model.train()

            optimizer.zero_grad()

            x, y = batch
            x, y = x.cuda(), y.cuda()

            x.requires_grad_(True)

            y_pred = model(x)

            y = F.one_hot(y, config.n_classes).float()

            loss = F.binary_cross_entropy(y_pred, y, reduction="mean")

            if config.training.duq.l_gradient_penalty > 0:
                gp = calc_gradient_penalty(x, y_pred)
                loss += config.training.duq.l_gradient_penalty * gp

            loss.backward()
            optimizer.step()

            x.requires_grad_(False)

            with torch.no_grad():
                model.eval()
                model.update_embeddings(x, y)

            return loss.item()

        def eval_step(engine, batch):
            model.eval()

            x, y = batch
            x, y = x.cuda(), y.cuda()

            x.requires_grad_(True)

            y_pred = model(x)

            return {"x": x, "y": y, "y_pred": y_pred}

        trainer = Engine(step)
        evaluator = Engine(eval_step)

        metric = Average()
        metric.attach(trainer, "loss")

        metric = Accuracy(output_transform=lambda out: (out["y_pred"], out["y"]))
        metric.attach(evaluator, "accuracy")

        def bce_output_transform(out):
            return (out["y_pred"], F.one_hot(out["y"], config.n_classes).float())

        metric = Loss(F.binary_cross_entropy, output_transform=bce_output_transform)
        metric.attach(evaluator, "bce")

        metric = Loss(
            lambda x, y: -F.binary_cross_entropy(x, y),
            output_transform=bce_output_transform,
        )
        metric.attach(evaluator, "neg_bce")

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(
                {"model": model},
                str(output_dir),
                n_saved=1,
                filename_prefix="best",
                score_name="neg_bce",
                global_step_transform=global_step_from_engine(trainer),
            ),
        )

        metric = Loss(
            calc_gradient_penalty,
            output_transform=lambda out: (out["x"], out["y_pred"]),
        )
        metric.attach(evaluator, "gradient_penalty")
        pbar = ProgressBar(dynamic_ncols=True)
        pbar.attach(trainer)

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_results(trainer):
            metrics = trainer.state.metrics
            loss = metrics["loss"]

            print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.2f}")

            writer.add_scalar("Loss/train", loss, trainer.state.epoch)
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            acc = metrics["accuracy"]
            bce = metrics["bce"]
            GP = metrics["gradient_penalty"]
            loss = bce + config.training.duq.l_gradient_penalty * GP

            print(
                (
                    f"Valid - Epoch: {trainer.state.epoch} "
                    f"Acc: {acc:.4f} "
                    f"Loss: {loss:.2f} "
                    f"BCE: {bce:.2f} "
                    f"GP: {GP:.2f} "
                )
            )

            writer.add_scalar("Loss/valid", loss, trainer.state.epoch)
            writer.add_scalar("BCE/valid", bce, trainer.state.epoch)
            writer.add_scalar("GP/valid", GP, trainer.state.epoch)
            writer.add_scalar("Accuracy/valid", acc, trainer.state.epoch)
            scheduler.step()

        trainer.run(train_loader, max_epochs=config.training.num_epochs)

        if trainer.state.epoch % 10 == 0:
            torch.save(
                model.state_dict(), f"{output_dir}/model-{trainer.state.epoch}.pt"
            )
        writer.close()


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
    main(args.config)
