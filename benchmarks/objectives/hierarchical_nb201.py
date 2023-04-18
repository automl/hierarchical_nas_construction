from __future__ import annotations

import os
import time
from copy import deepcopy

import torch
from neps.search_spaces.search_space import SearchSpace
from path import Path
from torch import nn
from torchvision import datasets as dset
from torchvision import transforms

from benchmarks.evaluation.objective import Objective
from benchmarks.objectives.custom_nb201.config_utils import load_config
from benchmarks.objectives.custom_nb201.custom_augmentations import (
    CUTOUT,
)
from benchmarks.objectives.custom_nb201.DownsampledImageNet import (
    ImageNet16,
)
from benchmarks.objectives.custom_nb201.evaluate_utils import (
    AverageMeter,
    get_optim_scheduler,
    obtain_accuracy,
    prepare_seed,
)

Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet-1k-s": 1000,
    "imagenet-1k": 1000,
    "ImageNet16": 1000,
    "ImageNet16-150": 150,
    "ImageNet16-120": 120,
    "ImageNet16-200": 200,
}


def get_dataset(
    name: str, root: str, cutout: int = -1, use_trivial_augment: bool = False
):
    if name == "cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == "cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith("imagenet-1k"):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif name.startswith("ImageNet16"):
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
    else:
        raise TypeError(f"Unknow dataset : {name}")

    # Data Argumentation
    if name == "cifar10" or name == "cifar100":
        if use_trivial_augment:
            raise NotImplementedError("Trivial augment impl. has to be added here!")
        else:
            lists = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if cutout > 0:
                lists += [CUTOUT(cutout)]
            train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
    elif name.startswith("ImageNet16"):
        if use_trivial_augment:
            raise NotImplementedError("Trivial augment impl. has to be added here!")
        else:
            lists = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(16, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            if cutout > 0:
                lists += [CUTOUT(cutout)]
            train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 16, 16)

    if name == "cifar10":
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == "cifar100":
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.startswith("imagenet-1k"):
        train_data = dset.ImageFolder(os.path.join(root, "train"), train_transform)
        test_data = dset.ImageFolder(os.path.join(root, "val"), test_transform)
        assert (
            len(train_data) == 1281167 and len(test_data) == 50000
        ), "invalid number of images : {:} & {:} vs {:} & {:}".format(
            len(train_data), len(test_data), 1281167, 50000
        )
    elif name == "ImageNet16":
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167 and len(test_data) == 50000
    elif name == "ImageNet16-120":
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700 and len(test_data) == 6000
    elif name == "ImageNet16-150":
        train_data = ImageNet16(root, True, train_transform, 150)
        test_data = ImageNet16(root, False, test_transform, 150)
        assert len(train_data) == 190272 and len(test_data) == 7500
    elif name == "ImageNet16-200":
        train_data = ImageNet16(root, True, train_transform, 200)
        test_data = ImageNet16(root, False, test_transform, 200)
        assert len(train_data) == 254775 and len(test_data) == 10000
    else:
        raise TypeError(f"Unknow dataset : {name}")

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_dataloaders(
    dataset: str,
    root: str,
    epochs: int,
    gradient_accumulations: int = 1,
    workers: int = 4,
    use_less: bool = False,
    use_trivial_augment: bool = False,
    eval_mode: bool = False,
):
    train_data, valid_data, xshape, class_num = get_dataset(
        name=dataset,
        root=root,
        cutout=-1,
        use_trivial_augment=use_trivial_augment,
    )
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    if dataset == "cifar10" or dataset == "cifar100":
        if use_less:  # check again
            config_path = "custom_nb201/configs/LESS.config"
        else:
            config_path = "custom_nb201/configs/CIFAR.config"
        split_info = load_config(dir_path / "custom_nb201/configs/cifar-split.txt", None)
    elif dataset.startswith("ImageNet16"):
        if use_less:
            config_path = "custom_nb201/configs/LESS.config"
        else:
            config_path = "custom_nb201/configs/ImageNet-16.config"
        split_info = load_config(
            dir_path / f"custom_nb201/configs/{dataset}-split.txt",
            None,
        )
    else:
        raise ValueError(f"invalid dataset : {dataset}")

    config = load_config(
        dir_path / config_path,
        {"class_num": class_num, "xshape": xshape, "epochs": epochs},
    )
    # check whether use splited validation set
    if dataset == "cifar10" and not eval_mode:
        ValLoaders = {
            "ori-test": torch.utils.data.DataLoader(
                valid_data,
                batch_size=config.batch_size // gradient_accumulations,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )
        }
        assert len(train_data) == len(split_info.train) + len(
            split_info.valid
        ), "invalid length : {:} vs {:} + {:}".format(
            len(train_data), len(split_info.train), len(split_info.valid)
        )
        train_data_v2 = deepcopy(train_data)
        train_data_v2.transform = valid_data.transform
        valid_data = train_data_v2
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size // gradient_accumulations,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.train),
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=config.batch_size // gradient_accumulations,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(split_info.valid),
            num_workers=workers,
            pin_memory=True,
        )
        ValLoaders["x-valid"] = valid_loader
    else:
        # data loader
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=config.batch_size // gradient_accumulations,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=config.batch_size // gradient_accumulations,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        if dataset == "cifar10":
            ValLoaders = {"ori-test": valid_loader}
        elif dataset == "cifar100":
            cifar100_splits = load_config(
                dir_path / "custom_nb201/configs/cifar100-test-split.txt", None
            )
            ValLoaders = {
                "ori-test": valid_loader,
                "x-valid": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        cifar100_splits.xvalid
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
                "x-test": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        cifar100_splits.xtest
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
            }
        elif dataset == "ImageNet16-120":
            imagenet16_splits = load_config(
                dir_path / "custom_nb201/configs/imagenet-16-120-test-split.txt", None
            )
            ValLoaders = {
                "ori-test": valid_loader,
                "x-valid": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        imagenet16_splits.xvalid
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
                "x-test": torch.utils.data.DataLoader(
                    valid_data,
                    batch_size=config.batch_size // gradient_accumulations,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        imagenet16_splits.xtest
                    ),
                    num_workers=workers,
                    pin_memory=True,
                ),
            }
        else:
            raise ValueError(f"invalid dataset : {dataset}")

    return config, train_loader, ValLoaders


def procedure(
    xloader,
    network,
    criterion,
    scheduler,
    optimizer,
    scaler,
    gradient_accumulations,
    mode,
):
    if mode == "train":
        network.train()
    elif mode == "valid":
        network.eval()
        top1, top5 = AverageMeter(), AverageMeter()
    else:
        raise ValueError(f"The mode is not right: {mode}")

    if mode == "train":
        network.zero_grad()
    for i, (inputs, targets) in enumerate(xloader):
        if mode == "train":
            scheduler.update(None, 1.0 * i / len(xloader))

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # forward
        with torch.cuda.amp.autocast():
            logits = network(inputs)
            loss = criterion(logits, targets)
        # backward
        if mode == "train":
            scaler.scale(loss / gradient_accumulations).backward()
            if (i + 1) % gradient_accumulations == 0:
                scaler.step(optimizer)
                scaler.update()
                network.zero_grad()
        # record loss and accuracy
        if mode == "valid":
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        # count time
    if mode == "valid":
        return top1.avg, top5.avg
    else:
        return None


def evaluate_for_seed(
    model: nn.Module,
    config,
    train_loader,
    valid_loaders: dict,
    gradient_accumulations,
    workers: int,
    working_directory: str | None = None,
    previous_working_directory: str | None = None,
):
    optimizer, scheduler, criterion = get_optim_scheduler(model.parameters(), config)
    scaler = torch.cuda.amp.GradScaler()
    if workers > 1:
        model = torch.nn.DataParallel(model)

    start_epoch = 0
    total_epochs = config.epochs + config.warmup
    if previous_working_directory is not None:
        checkpoint = torch.load(
            os.path.join(previous_working_directory, "checkpoint.pth")
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epochs_trained"]

    model, criterion = model.cuda(), criterion.cuda()
    for epoch in range(start_epoch, total_epochs):
        scheduler.update(epoch, 0.0)
        _ = procedure(
            xloader=train_loader,
            network=model,
            criterion=criterion,
            scheduler=scheduler,
            optimizer=optimizer,
            scaler=scaler,
            gradient_accumulations=gradient_accumulations,
            mode="train",
        )

    with torch.no_grad():
        out_dict = {}
        for key, xloader in valid_loaders.items():
            valid_acc1, valid_acc5 = procedure(
                xloader=xloader,
                network=model,
                criterion=criterion,
                scheduler=None,
                optimizer=None,
                scaler=None,
                gradient_accumulations=None,
                mode="valid",
            )
            out_dict[f"{key}_1"] = valid_acc1
            out_dict[f"{key}_5"] = valid_acc5

    if working_directory is not None:
        model.train()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epochs_trained": total_epochs,
            },
            os.path.join(working_directory, "checkpoint.pth"),
            _use_new_zipfile_serialization=False,
        )

    del model
    del criterion
    del scheduler
    del optimizer

    return out_dict


class NB201Pipeline(Objective):
    n_epochs = 200
    gradient_accumulations = 1  # 2
    workers = 4

    def __init__(
        self,
        dataset: str,
        data_path,
        seed: int,
        log_scale: bool = True,
        negative: bool = False,
        eval_mode: bool = False,
        is_fidelity: bool = False,
    ) -> None:
        assert seed in [555, 666, 777, 888, 999]
        super().__init__(seed, log_scale, negative)
        self.dataset = dataset
        self.data_path = data_path
        self.failed_runs = 0

        self.eval_mode = eval_mode
        if self.eval_mode:
            self.n_epochs = 200

        self.is_fidelity = is_fidelity

        if self.dataset == "cifar10":
            self.num_classes = 10
        elif self.dataset == "cifar100":
            self.num_classes = 100
        elif self.dataset == "ImageNet16-120":
            self.num_classes = 120
        else:
            raise NotImplementedError

    def __call__(self, working_directory, previous_working_directory, architecture, **hp):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gradient_accumulations = self.gradient_accumulations
        while gradient_accumulations < 16:
            try:
                start = time.time()
                config, train_loader, ValLoaders = get_dataloaders(
                    self.dataset,
                    self.data_path,
                    epochs=self.n_epochs,
                    gradient_accumulations=gradient_accumulations,
                    workers=self.workers,
                    use_trivial_augment=hp["trivial_augment"]
                    if "trivial_augment" in hp
                    else False,
                    eval_mode=self.eval_mode,
                )
                prepare_seed(self.seed, self.workers)
                if hasattr(architecture, "to_pytorch"):
                    model = architecture.to_pytorch()
                else:
                    assert isinstance(architecture, nn.Module)
                    model = architecture

                if self.is_fidelity:
                    out_dict = evaluate_for_seed(
                        model,
                        config,
                        train_loader,
                        ValLoaders,
                        gradient_accumulations=gradient_accumulations,
                        workers=self.workers,
                        working_directory=working_directory,
                        previous_working_directory=previous_working_directory,
                    )
                else:
                    out_dict = evaluate_for_seed(
                        model,
                        config,
                        train_loader,
                        ValLoaders,
                        gradient_accumulations=gradient_accumulations,
                        workers=self.workers,
                    )
                end = time.time()
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    gradient_accumulations *= 2
                    torch.cuda.empty_cache()
                else:
                    raise e

        # prepare result_dict
        if self.eval_mode and "x-valid_1" not in out_dict:
            val_err = 1
        else:
            val_err = 1 - out_dict["x-valid_1"] / 100

        nof_parameters = sum(p.numel() for p in model.parameters())
        results = {
            "loss": self.transform(val_err),
            "info_dict": {
                **out_dict,
                **{
                    "train_time": end - start,
                    "timestamp": end,
                    "number_of_parameters": nof_parameters,
                },
            },
        }

        del model
        del train_loader
        del ValLoaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def get_train_loader(self):
        _, train_loader, _ = get_dataloaders(
            self.dataset,
            self.data_path,
            epochs=self.n_epochs,
            gradient_accumulations=1,
            workers=self.workers,
            use_trivial_augment=False,
            eval_mode=self.eval_mode,
        )
        return train_loader

if __name__ == "__main__":
    import argparse
    import re
    from functools import partial

    # pylint: disable=ungrouped-imports
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from benchmarks.objectives.apis.nasbench201 import NAS201
    from benchmarks.objectives.custom_nb201.genotypes import (
        Structure as CellStructure,
    )
    from benchmarks.objectives.custom_nb201.tiny_network import (
        TinyNetwork,
    )
    from benchmarks.objectives.nasbench201 import NasBench201Objective
    from benchmarks.search_spaces.hierarchical_nb201.graph import (
        NB201Spaces,
    )

    # pylint: enable=ungrouped-imports

    def convert_identifier_to_str(identifier: str, terminals_to_nb201: dict) -> str:
        """
        Converts identifier to string representation.
        """
        start_indices = [m.start() for m in re.finditer("(OPS*)", identifier)]
        op_edge_list = []
        counter = 0
        for i, _ in enumerate(start_indices):
            start_idx = start_indices[i]
            end_idx = start_indices[i + 1] if i < len(start_indices) - 1 else -1
            substring = identifier[start_idx:end_idx]
            for k in terminals_to_nb201.keys():
                if k in substring:
                    op_edge_list.append(f"{terminals_to_nb201[k]}~{counter}")
                    break
            if i == 0 or i == 2:
                counter = 0
            else:
                counter += 1

        return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--dataset",
        default="cifar-10",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        default="",
        help="Path to folder with data or where data should be saved to if downloaded.",
        type=str,
    )
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--write_graph", action="store_true")
    parser.add_argument("--best_archs", action="store_true")
    parser.add_argument("--nb201_model_backend", action="store_true")
    args = parser.parse_args()

    pipeline_space = dict(
        architecture=NB201Spaces(
            space="variable_multi_multi",
            dataset="cifar10",
            use_prior=True,
            adjust_params=False,
        ),
    )
    pipeline_space = SearchSpace(**pipeline_space)
    pipeline_space = pipeline_space.sample(user_priors=True)
    run_pipeline_fn = NB201Pipeline(
        dataset=args.dataset, data_path=args.data_path, seed=args.seed, eval_mode=True
    )
    res = run_pipeline_fn("", "", pipeline_space.hyperparameters["architecture"])

    pipeline_space = dict(
        architecture=NB201Spaces(
            space="fixed_1_none", dataset=args.dataset, adjust_params=False
        ),
    )
    pipeline_space = SearchSpace(**pipeline_space)
    sampled_pipeline_space = pipeline_space.sample()

    # cell_shared = original NB201 space
    pipeline_space = dict(
        architecture=NB201Spaces(
            space="variable_multi_multi", dataset=args.dataset, adjust_params=False
        ),
    )
    pipeline_space = SearchSpace(**pipeline_space)
    sampled_pipeline_space = pipeline_space.sample()
    identifier = {
        "architecture": "(D2 Linear3 (D1 Linear3 (C Diamond2 (CELL Cell (OPS zero) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv1x1o) (NORM layer))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV conv3x3o) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV dconv3x3o) (NORM layer)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV conv1x1o) (NORM instance))) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV conv1x1o) (NORM instance))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT mish) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV dconv3x3o) (NORM batch)))) (CELL Cell (OPS zero) (OPS zero) (OPS zero) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV dconv3x3o) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM layer) (ACT relu)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM batch) (ACT relu))) (OPS zero) (OPS avg_pool) (OPS zero) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT mish))))) (C Diamond2 (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM instance) (ACT hardswish))) (OPS zero) (OPS zero) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV dconv3x3o) (NORM layer))) (OPS zero)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT hardswish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM batch) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV dconv3x3o) (NORM layer))) (OPS id)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT hardswish) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (ACT relu) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV dconv3x3o) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT relu) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT hardswish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM instance) (ACT mish)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM layer) (ACT mish))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM batch))) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV conv3x3o) (NORM layer))) (OPS avg_pool) (OPS id) (OPS zero))) (DOWN Residual2 (CELL Cell (OPS zero) (OPS zero) (OPS id) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT hardswish) (NORM batch))) (OPS zero)) resBlock resBlock)) (D1 Linear3 (C Linear2 (CELL Cell (OPS avg_pool) (OPS avg_pool) (OPS avg_pool) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT relu))) (OPS zero)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT mish))) (OPS id) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM batch) (ACT relu))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT mish))) (OPS avg_pool))) (C Linear2 (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM instance) (ACT hardswish))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (ACT mish) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM layer) (ACT relu))) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM batch) (ACT relu))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (ACT mish) (NORM layer)))) (CELL Cell (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT hardswish))) (OPS id) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM layer) (ACT hardswish))) (OPS zero))) (DOWN Residual2 (CELL Cell (OPS zero) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT relu) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT relu) (NORM layer))) (OPS zero) (OPS avg_pool)) resBlock resBlock)) (D0 Residual3 (C Residual2 (CELL Cell (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv1x1o) (NORM batch))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV dconv3x3o) (NORM instance))) (OPS id) (OPS avg_pool)) (CELL Cell (OPS avg_pool) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (ACT hardswish) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM batch) (ACT relu))) (OPS avg_pool) (OPS avg_pool)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM layer) (ACT hardswish))) (OPS avg_pool) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT hardswish) (CONV conv3x3o) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM instance) (ACT mish))) (OPS zero))) (C Residual2 (CELL Cell (OPS avg_pool) (OPS avg_pool) (OPS avg_pool) (OPS zero) (OPS zero) (OPS avg_pool)) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM layer))) (OPS avg_pool) (OPS avg_pool) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM layer))) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT mish) (NORM instance)))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT hardswish))) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT mish) (NORM batch))) (OPS zero) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv3x3o) (NORM instance) (ACT relu))) (OPS id))) (CELL Cell (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (ACT hardswish) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (CONV dconv3x3o) (NORM batch) (ACT hardswish))) (OPS id) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV conv3x3o) (NORM instance))) (OPS Linear1 (CONVBLOCK Linear3 (ACT relu) (CONV dconv3x3o) (NORM instance))) (OPS zero)) (CELL Cell (OPS zero) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (CONV conv1x1o) (NORM layer) (ACT hardswish))) (OPS avg_pool) (OPS Linear1 (CONVBLOCK Linear3 (ACT mish) (CONV conv3x3o) (NORM instance))) (OPS zero))))"
    }
    sampled_pipeline_space.load_from(identifier)
    run_pipeline_fn = NB201Pipeline(
        dataset=args.dataset, data_path=args.data_path, seed=args.seed, eval_mode=True
    )
    res = run_pipeline_fn("", "", sampled_pipeline_space.hyperparameters["architecture"])

    if args.write_graph:
        writer = SummaryWriter("results/hierarchical_nb201")
        net = sampled_pipeline_space.hyperparameters["architecture"].to_pytorch()
        images = torch.randn((8, 3, 32, 32))
        _ = sampled_pipeline_space.hyperparameters["architecture"](images)
        _ = net(images)
        writer.add_graph(net, images)
        writer.close()

    terminals_to_nb201 = {
        "avg_pool": "avg_pool_3x3",
        "conv1x1": "nor_conv_1x1",
        "conv3x3": "nor_conv_3x3",
        "id": "skip_connect",
        "zero": "none",
    }
    identifier_to_str_mapping = partial(
        convert_identifier_to_str, terminals_to_nb201=terminals_to_nb201
    )
    api = NAS201(
        os.path.dirname(args.data_path),
        negative=False,
        seed=args.seed,
        task=f"{args.dataset}-valid" if args.dataset == "cifar10" else args.dataset,
        log_scale=True,
        identifier_to_str_mapping=identifier_to_str_mapping,
    )
    run_pipeline_fn = NasBench201Objective(api)

    if args.best_archs:
        generator = (
            sampled_pipeline_space.hyperparameters["architecture"].grammars[0].generate()
        )
        archs = list(generator)
        identifier = sampled_pipeline_space.serialize()["architecture"]
        vals = {}
        for arch in tqdm(archs):
            start_idx = 0
            new_identifier = identifier
            for ops in arch[1:]:
                starting_idx = new_identifier.find("OPS", start_idx)
                empty_idx = new_identifier.find(" ", starting_idx)
                closing_idx = new_identifier.find(")", starting_idx)
                new_identifier = (
                    new_identifier[: empty_idx + 1] + ops + new_identifier[closing_idx:]
                )
                start_idx = new_identifier.find(")", starting_idx)

            try:
                sampled_pipeline_space.load_from({"architecture": new_identifier})
                res_api = run_pipeline_fn(
                    sampled_pipeline_space.hyperparameters["architecture"]
                )
                vals[new_identifier] = 100 * (1 - res_api["info_dict"]["val_score"])
            except Exception:
                pass

        results = sorted(vals.items(), key=lambda pair: pair[1], reverse=True)[:10]
        print(args.seed)
        print(results)
    else:
        # seed 777
        identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS conv3x3) (OPS id) (OPS conv3x3) (OPS conv1x1))"
        # identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS zero) (OPS id) (OPS conv3x3) (OPS conv1x1))"
        # identifier = "(CELL Cell (OPS conv3x3) (OPS conv3x3) (OPS avg_pool) (OPS id) (OPS conv3x3) (OPS conv3x3))"
        if args.nb201_model_backend:
            nb201_identifier = identifier_to_str_mapping(identifier)
            genotype = CellStructure.str2structure(nb201_identifier)
            arch_config = {"channel": 16, "num_cells": 5}
            if args.dataset == "cifar10":
                n_classes = 10
            elif args.dataset == "cifar100":
                n_classes = 100
            elif args.dataset == "ImageNet16-120":
                n_classes = 120
            else:
                raise NotImplementedError
            tiny_net = TinyNetwork(
                arch_config["channel"], arch_config["num_cells"], genotype, n_classes
            )

        sampled_pipeline_space.load_from({"architecture": identifier})

        if args.nb201_model_backend:
            our_model = sampled_pipeline_space.hyperparameters[
                "architecture"
            ].to_pytorch()

            tiny_net_total_params = sum(p.numel() for p in tiny_net.parameters())
            our_model_total_params = sum(p.numel() for p in our_model.parameters())
            print(tiny_net_total_params, our_model_total_params)

            new_state_dict = {k: None for k in our_model.state_dict().keys()}
            our_model_values = our_model.state_dict().values()
            for (k_tiny, v_tiny), (k_our, v_our) in zip(
                tiny_net.state_dict().items(), our_model.state_dict().items()
            ):
                new_state_dict[k_our] = v_tiny
            our_model.load_state_dict(new_state_dict)

            input_img = torch.randn((8, 3, 32, 32))
            output_tiny = tiny_net(input_img)
            output_our = our_model(input_img)
            print(
                f"Model is functionally equivalent: {torch.all(output_tiny == output_our)}"
            )

        res_api = run_pipeline_fn(sampled_pipeline_space.hyperparameters["architecture"])
        res_api_val = 100 * (1 - res_api["info_dict"]["val_score"])
        res_api_test = 100 * (1 - res_api["info_dict"]["test_score"])
        del run_pipeline_fn
        del api

        run_pipeline_fn = NB201Pipeline(
            dataset=args.dataset, data_path=args.data_path, seed=args.seed, eval_mode=True
        )
        if args.nb201_model_backend:
            res = run_pipeline_fn("", "", tiny_net)
        else:
            res = run_pipeline_fn(
                "", "", sampled_pipeline_space.hyperparameters["architecture"]
            )
        res_val = res["info_dict"]["valid_acc1es"][-1]
        res_test = res["info_dict"]["test_acc1es"][-1]

        print(args.seed)
        print(f"Id: {sampled_pipeline_space.serialize()}")
        print("\t\tAPI\t\t\tRun")
        print(f"Val\t{res_api_val}\t{res_val}")
        print(f"Test\t{res_api_test}\t{res_test}")
