import time
from typing import Union

import numpy as np
import torch
from path import Path

from benchmarks.evaluation.objective import Objective
from benchmarks.objectives.darts_utils.train import train_evaluation
from benchmarks.objectives.darts_utils.train_search import train_search
from benchmarks.search_spaces.darts_cnn.genotypes import Genotype


class DARTSCnn(Objective):
    def __init__(
        self,
        data_path: Union[str, Path],
        eval_policy: str = "last5",
        seed: int = 777,
        log_scale: bool = True,
        negative: bool = False,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(seed, log_scale, negative)
        assert eval_policy in ["best", "last", "last5"]

        self.data_path = data_path
        self.eval_policy = eval_policy
        self.eval_mode = eval_mode

    def __call__(self, working_directory, previous_working_directory, normal, reduce):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(normal, "to_pytorch"):
            normal = normal.to_pytorch()
        else:
            raise NotImplementedError
        if hasattr(reduce, "to_pytorch"):
            reduce = reduce.to_pytorch()
        else:
            raise NotImplementedError

        genotype = Genotype(
            normal=normal,
            normal_concat=range(2, 6),
            reduce=reduce,
            reduce_concat=range(2, 6),
        )
        start = time.time()
        if self.eval_mode:
            valid_accs = train_evaluation(
                genotype=genotype,
                data=self.data_path,
                seed=self.seed,
                save_path=working_directory,
            )
        else:
            valid_accs = train_search(
                genotype=genotype,
                data=self.data_path,
                seed=self.seed,
                save_path=working_directory,
            )
        end = time.time()

        if "best" == self.eval_policy:
            val_error = 1 - max(valid_accs) / 100
        elif "last" == self.eval_policy:
            val_error = 1 - valid_accs[-1] / 100
        elif "last5" == self.eval_policy:
            val_error = 1 - np.mean(valid_accs[-5:]) / 100

        return {
            "loss": self.transform(val_error),
            "info_dict": {
                "accs": valid_accs,
                "best_acc": max(valid_accs),
                "last_acc": valid_accs[-1],
                "time": end - start,
                "timestamp": end,
            },
        }


if __name__ == "__main__":
    import argparse
    import json
    import os
    import shutil

    import yaml
    from neps.search_spaces.search_space import SearchSpace

    # pylint: disable=ungrouped-imports
    from benchmarks.search_spaces.darts_cnn.graph import DARTSSpace

    # pylint: enable=ungrouped-imports

    parser = argparse.ArgumentParser(description="Train DARTS")
    parser.add_argument(
        "--data",
        default="",
        help="Path to folder with data or where data should be saved to if downloaded.",
        type=str,
    )
    parser.add_argument(
        "--save",
        default="",
        type=str,
    )
    parser.add_argument(
        "--arch", default="ours", type=str, choices=["ours", "drnas", "nasbowl"]
    )
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    pipeline_space = dict(
        normal=DARTSSpace(),
        reduce=DARTSSpace(),
    )
    pipeline_space = SearchSpace(**pipeline_space)

    run_pipeline_fn = DARTSCnn(data_path=args.data, seed=args.seed, eval_mode=args.eval)

    if args.arch == "drnas":
        # DrNAS cells
        pipeline_space.load_from(
            {
                "normal": "(CELL DARTS (OP sep_conv_3x3) (IN1 0) (OP sep_conv_5x5) (IN1 1) (OP sep_conv_3x3) (IN2 1) (OP sep_conv_3x3) (IN2 2) (OP skip_connect) (IN3 0) (OP sep_conv_3x3) (IN3 1) (OP sep_conv_3x3) (IN4 2) (OP dil_conv_5x5) (IN4 3))",
                "reduce": "(CELL DARTS (OP max_pool_3x3) (IN1 0) (OP sep_conv_5x5) (IN1 1) (OP dil_conv_5x5) (IN2 2) (OP sep_conv_5x5) (IN2 1) (OP sep_conv_5x5) (IN3 1) (OP dil_conv_5x5) (IN3 3) (OP skip_connect) (IN4 4) (OP sep_conv_5x5) (IN4 1))",
            }
        )
        working_directory = Path(args.save) / "drnas"
    elif args.arch == "nasbowl":
        # nasbowl cells
        pipeline_space.load_from(
            {
                "normal": "(CELL DARTS (OP skip_connect) (IN1 1) (OP sep_conv_3x3) (IN1 0) (OP sep_conv_3x3) (IN2 1) (OP max_pool_3x3) (IN2 0) (OP sep_conv_5x5) (IN3 1) (OP sep_conv_3x3) (IN3 0) (OP dil_conv_5x5) (IN4 2) (OP sep_conv_3x3) (IN4 1))",
                "reduce": "(CELL DARTS (OP skip_connect) (IN1 1) (OP sep_conv_3x3) (IN1 0) (OP sep_conv_3x3) (IN2 1) (OP max_pool_3x3) (IN2 0) (OP sep_conv_5x5) (IN3 1) (OP sep_conv_3x3) (IN3 0) (OP dil_conv_5x5) (IN4 2) (OP sep_conv_3x3) (IN4 1))",
            }
        )
        working_directory = Path(args.save) / "bananas"
    elif args.arch == "ours":
        args.save = Path(args.save)
        results_dir = args.save / "results"
        assert os.path.isdir(results_dir)
        config_loss_dict = {}
        for config_number in os.listdir(results_dir):
            results_yaml = results_dir / config_number / "result.yaml"
            if os.path.isfile(results_yaml):
                with open(results_yaml) as f:
                    data = yaml.safe_load(f)
                config_loss_dict[config_number] = data["loss"]
        best_config = min(config_loss_dict, key=config_loss_dict.get)
        config_yaml = results_dir / best_config / "config.yaml"
        with open(config_yaml) as f:
            identifier = yaml.safe_load(f)
        pipeline_space.load_from(identifier)
        working_directory = args.save / f"best_config_eseed_{args.seed}"
        working_directory.makedirs_p()
        shutil.copyfile(config_yaml, working_directory / "config.yaml")
    else:
        raise NotImplementedError
    working_directory.makedirs_p()

    res = run_pipeline_fn(
        working_directory,
        "",
        pipeline_space.hyperparameters["normal"],
        pipeline_space.hyperparameters["reduce"],
    )
    print(args.arch, res)

    with open(
        working_directory / "results.json"
        if args.arch == "ours"
        else f"{args.arch}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=4)
