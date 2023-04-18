import time

import torch
from neps.search_spaces.search_space import SearchSpace

from benchmarks.evaluation.objective import Objective
from benchmarks.evaluation.train import training_pipeline
from benchmarks.evaluation.utils import (
    get_evaluation_metric,
    get_loss,
    get_optimizer,
    get_scheduler,
    get_train_val_test_loaders,
)


class AddNISTObjective(Objective):
    dataset = "addNIST"
    n_epochs = 64
    batch_size = 64
    optim_kwargs = {"lr": 0.01, "momentum": 0.9, "weight_decay": 3e-4}
    num_classes = 20

    def __init__(
        self,
        data_path,
        seed,
        log_scale: bool = True,
        negative: bool = False,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(seed, log_scale, negative)
        self.data_path = data_path
        self.failed_runs = 0

        self.eval_mode = eval_mode
        if self.eval_mode:
            self.n_epochs = 64

    def __call__(self, working_directory, previous_working_director, architecture, **hp):
        start = time.time()
        if isinstance(architecture, SearchSpace):
            model = architecture.hyperparameters["graph"].get_model_for_evaluation()
            for key in self.optim_kwargs:
                if key in architecture.hyperparameters:
                    self.optim_kwargs[key] = architecture.hyperparameters[key].value
        elif hasattr(architecture, "get_model_for_evaluation"):
            model = architecture.get_model_for_evaluation()
        elif hasattr(architecture, "to_pytorch"):
            model = architecture.to_pytorch()
        elif isinstance(architecture, torch.nn.Module):  # assumes to be a PyTorch model
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
        try:
            if not self.eval_mode:
                val_err = 1 - results["val_scores"][-1]
        except Exception as e:
            print(e)
            val_err = 1.0
            self.failed_runs += 1
            if self.failed_runs > 10:
                raise Exception("Too many failed runs!")
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
        if self.eval_mode:
            results["train_time"] = end - start
            return results
        if isinstance(architecture, SearchSpace):
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
