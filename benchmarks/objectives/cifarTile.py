import random
import time

import numpy as np
import torch

from benchmarks.evaluation.objective import Objective
from benchmarks.evaluation.train import training_pipeline
from benchmarks.evaluation.utils import (
    get_evaluation_metric,
    get_loss,
    get_optimizer,
    get_scheduler,
    get_train_val_test_loaders,
)


def prepare_seed(rand_seed: int, workers: int = 4):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.set_num_threads(workers)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


class CifarTileObjective(Objective):
    dataset = "cifarTile"
    batch_size = 64
    optim_kwargs = {"lr": 0.01, "momentum": 0.9, "weight_decay": 3e-4}
    n_epochs = 64
    workers = 2
    num_classes = 4

    def __init__(
        self,
        data_path,
        seed: int,
        log_scale: bool = True,
        negative: bool = False,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(seed, log_scale, negative)
        self.data_path = data_path
        self.eval_mode = eval_mode

    def __call__(self, working_directory, previous_working_director, architecture, **kwargs):
        start = time.time()

        prepare_seed(self.seed, self.workers)

        if hasattr(architecture, "to_pytorch"):
            model = architecture.to_pytorch()
        elif isinstance(architecture, torch.nn.Module):
            model = architecture
        else:
            raise NotImplementedError

        model.cuda()
        model.train()
        train_criterion = get_loss("CrossEntropyLoss")
        evaluation_metric = get_evaluation_metric("Accuracy", top_k=1)
        evaluation_metric.cuda()

        optimizer = get_optimizer("SGD", model, **self.optim_kwargs)
        scheduler = get_scheduler(
            scheduler="CosineAnnealingLR", optimizer=optimizer, T_max=self.n_epochs
        )
        train_loader, valid_loader, test_loader = get_train_val_test_loaders(
            dataset=self.dataset,
            data=self.data_path,
            batch_size=self.batch_size,
            eval_mode=self.eval_mode,
        )
        results = training_pipeline(
            model=model,
            train_criterion=train_criterion,
            evaluation_metric=evaluation_metric,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            n_epochs=self.n_epochs,
            eval_mode=self.eval_mode,
        )
        end = time.time()
        del model
        del train_criterion
        del evaluation_metric
        del optimizer
        del scheduler
        del train_loader
        del valid_loader
        del test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_err = 1 - results["val_scores"][-1]
        return {
            "loss": self.transform(val_err),
            "info_dict": {
                "val_score": val_err,
                "val_scores": results["val_scores"],
                "test_score": 1 - results["test_scores"][-1],
                "test_scores": results["test_scores"],
                "train_time": end - start,
                "timestamp": end,
            },
        }

    def get_train_loader(self):
        train_loader, _, _ = get_train_val_test_loaders(
            dataset=self.dataset,
            data=self.data_path,
            batch_size=self.batch_size,
            eval_mode=self.eval_mode,
        )
        return train_loader

if __name__ == "__main__":
    import argparse

    # pylint: disable=ungrouped-imports
    from neps.search_spaces.search_space import SearchSpace

    from benchmarks.search_spaces.hierarchical_nb201.graph import (
        NB201Spaces,
    )

    # pylint: enable=ungrouped-imports

    parser = argparse.ArgumentParser(description="Train CifarTile")
    parser.add_argument(
        "--data_path",
        help="Path to folder with data or where data should be saved to if downloaded.",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="cifarTile",
        type=str,
    )
    parser.add_argument("--seed", default=777, type=int)
    args = parser.parse_args()

    pipeline_space = dict(
        architecture=NB201Spaces(
            space="variable_multi_multi", dataset=args.dataset, adjust_params=False
        )
    )
    pipeline_space = SearchSpace(**pipeline_space)
    pipeline_space = pipeline_space.sample()

    run_pipeline_fn = CifarTileObjective(data_path=args.data_path, seed=args.seed)
    res = run_pipeline_fn(architecture=pipeline_space.hyperparameters["architecture"])
