import torch
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import time
from torchvision import transforms, datasets
from torch.cuda.amp import GradScaler
from copy import deepcopy

from benchmarks.evaluation.objective import Objective

from benchmarks.objectives.custom_nb201.evaluate_utils import (
    AverageMeter,
    obtain_accuracy,
    prepare_seed,
)

def get_dataloaders(dataset, data_path, workers, seed: int = 777, eval_mode: bool = False):
    if dataset == "cifar10":
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        val_test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        train_set = datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=True)
        val_set = datasets.CIFAR10(root=data_path, train=True, transform=val_test_transform, download=True)

        num_train = len(train_set)
        val_split = []
        class_dict = {i: 0 for i in range(10)}
        counter = 0
        while len(val_split) < 5000 and counter < num_train:
            label = train_set[counter][1]
            if class_dict[label] < 500:
                class_dict[label] += 1
                val_split.append(counter)
            counter += 1
        train_idx, val_idx = [i for i in range(num_train) if i not in val_split], val_split
        assert set(train_idx).intersection(val_split) == set() and len(train_idx) + len(val_idx) == num_train
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        if not eval_mode:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=128,
                sampler=train_sampler,
                num_workers=workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=128, sampler=val_sampler,
                num_workers=workers, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=128,
                shuffle=True,
                num_workers=workers, pin_memory=True)
            val_loader = None

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_path, train=False, transform=val_test_transform, download=True),
            batch_size=128, shuffle=False,
            num_workers=workers, pin_memory=True)
    else:
        raise NotImplementedError

    return train_loader, val_loader, test_loader


@torch.no_grad()
def eval_val_test(model, loader):
    model.eval()
    top1_error = AverageMeter()
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = obtain_accuracy(outputs, targets, topk=(1,))[0]
        err1 = 1 - acc1 / 100
        top1_error.update(err1.item(), outputs.size(0))
    return top1_error.avg


def evaluate(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, n_steps: int, eval: bool = False):
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)
    scaler = GradScaler()

    c_step = 0
    model.train()
    early_stop = False
    while c_step < n_steps and not early_stop:
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # # early stopping criterion if activation function does not work
            if not eval and c_step > 0 and c_step == 10000 and val_loader is not None:
                model.eval()
                val_error = eval_val_test(model, val_loader)
                if val_error > 0.4:
                    early_stop = True
                    break
                model.train()

            if c_step >= n_steps:
                break

            c_step += 1

    model.eval()
    if not early_stop:
        if val_loader is not None:
            val_error = eval_val_test(model, val_loader)
        else:
            val_error = 1.

    if test_loader is not None:
        test_error = eval_val_test(model, test_loader)
    else:
        test_error = 1.

    return {
        "val_error": val_error,
        "test_error": test_error,
        "early_stop": early_stop,
    }


class CIFAR10ActivationObjective(Objective):
    n_steps = 64000
    workers = 4
    __name__ = "CIFAR10ActivationObjective"

    def __init__(
        self,
        dataset: str,
        data_path,
        seed: int,
        log_scale: bool = True,
        negative: bool = False,
        eval_mode: bool = False,
    ) -> None:
        assert seed in [555, 666, 777, 888, 999]
        super().__init__(seed, log_scale, negative)
        self.dataset = dataset
        self.data_path = data_path
        self.failed_runs = 0

        self.eval_mode = eval_mode
        if not self.eval_mode:
            pass

    def __call__(self, working_directory, previous_working_directory, architecture, **hp):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = time.time()
        prepare_seed(self.seed, self.workers)
        train_loader, val_loader, test_loader = get_dataloaders(
            self.dataset,
            self.data_path,
            workers=self.workers,
            eval_mode=self.eval_mode,
        )
        if hasattr(architecture, "to_pytorch"):
            model = architecture.to_pytorch()
        else:
            assert isinstance(architecture, nn.Module)
            model = architecture

        if not self.eval_mode:
            try:
                out_dict = evaluate(
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    n_steps = self.n_steps,
                    eval = self.eval_mode,
                )
            except Exception:
                out_dict = {"val_error": 1., "test_error":1.}
        else:
            out_dict = evaluate(
                model,
                train_loader,
                val_loader,
                test_loader,
                n_steps = self.n_steps,
                eval = self.eval_mode,
            )
        end = time.time()

        # prepare result_dict
        val_err = out_dict["val_error"] if not self.eval_mode else 1.
        results = {
            "loss": self.transform(val_err),
            "info_dict": {
                **out_dict,
                **{
                    "train_time": end - start,
                    "timestamp": end,
                },
            },
        }

        del model
        del train_loader
        del val_loader
        del test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

def dummy_test(config, is_trainable=True, is_const: bool = False):
    model = config["architecture"].to_pytorch()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_trainable:
        assert pytorch_total_params > 269722
        first_act_before = deepcopy([p.data for p in model.act.parameters() if p.requires_grad][0].flatten())
        second_act_before = deepcopy([p.data for p in model.layer1[0].act1.parameters() if p.requires_grad][0].flatten())

    model.train()
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    inputs = torch.rand((128, 3, 32, 32))
    targets = torch.rand((128, 10))
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if is_trainable and not is_const:
        first_act_after = [p.data for p in model.act.parameters() if p.requires_grad][0].flatten()
        second_act_after = [p.data for p in model.layer1[0].act1.parameters() if p.requires_grad][0].flatten()
        assert not torch.all(first_act_after == first_act_before)
        assert not torch.all(second_act_after == second_act_before)
        assert not torch.all(first_act_after == second_act_after)

    model.eval()
    outputs = model(inputs)

if __name__ == "__main__":
    import argparse
    import math
    import json
    from path import Path
    from neps.search_spaces.search_space import SearchSpace
    from benchmarks.search_spaces.activation_function_search.graph import ActivationSpace
    import benchmarks.search_spaces.activation_function_search.cifar_models as cifar_models

    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--save_path",
        default="tmp/",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        default="",
        help="Path to folder with data or where data should be saved to if downloaded.",
        type=str,
    )
    parser.add_argument(
        "--base_architecture",
        default="resnet20",
        type=str,
    )
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument("--resnet20", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    args = parser.parse_args()

    if not args.DEBUG:
        args.save_path = Path(args.save_path)
        args.save_path.makedirs_p()

    if args.resnet20:
        model = getattr(cifar_models, "resnet20")(num_classes=10)
    else:
        pipeline_space = dict(
            architecture=ActivationSpace(
                dataset=args.dataset,
                base_architecture=args.base_architecture,
            ),
        )
        pipeline_space = SearchSpace(**pipeline_space)
        print(
            "benchmark",
            math.log10(pipeline_space.hyperparameters["architecture"].search_space_size),
        )

        # unary_ops = ["id", "neg", "abs", "square", "cubic", "square_root", "mconst", "aconst", "log", "exp", "sin", "cos", "sinh", "cosh", "tanh", "asinh", "atanh", "sinc", "umax", "umin", "sigmoid", "logexp", "gaussian", "erf", "const"]
        # for unary_op in unary_ops:
        #     print(unary_op)
        #     unary_id_test = {"architecture": f'(L3 UnaryTopo (L2 UnaryTopo (UnOp {unary_op})))'}
        #     pipeline_space.load_from(unary_id_test)
        #     dummy_test(pipeline_space, PRIMITIVES[unary_op].trainable, is_const="const"==unary_op)

        # binary_ops = ["add", "multi", "sub", "div", "bmax", "bmin", "bsigmoid", "bgaussian_sq", "bgaussian_abs", "wavg"]
        # for binary_op in binary_ops:
        #     print(binary_op)
        #     binary_id_test = {"architecture": f'(L3 BinaryTopo (BinOp {binary_op}) (L2 UnaryTopo (UnOp sin)) (L2 UnaryTopo (UnOp erf)) id id)'}
        #     pipeline_space.load_from(binary_id_test)
        #     dummy_test(pipeline_space, PRIMITIVES[binary_op].trainable)

        # prepare a single model for full training
        pipeline_space = pipeline_space.sample()
        model = pipeline_space.hyperparameters["architecture"]

        #id = "(L2 BinaryTopo (BinOp bsigmoid) (L1 UnaryTopo (UnOp mconst)) (L1 UnaryTopo (UnOp id)) id id)" # swish
        # id = "(L2 UnaryTopo (L1 UnaryTopo (UnOp umax)))" # relu
        id = "(L2 BinaryTopo (BinOp wavg) (L1 BinaryTopo (BinOp bsigmoid) (UnOp neg) (UnOp umin) id id) (L1 BinaryTopo (BinOp bmin) (UnOp umax) (UnOp erf) id id) id id)"
        pipeline_space.load_from({
            "architecture": id
        })
        model = pipeline_space.hyperparameters["architecture"]
        print(id)

    run_pipeline_fn = CIFAR10ActivationObjective(
        dataset=args.dataset, data_path=args.data_path, seed=args.seed, eval_mode=args.eval
    )
    res = run_pipeline_fn("", "", model)
    print("eval mode" if args.eval else "search mode", args.base_architecture, res)

    results = {
        "id": id,
        "res": res,
        "mode": "eval" if args.eval else "search",
        "seed": args.seed,
    }
    if not args.DEBUG:
        with open(args.save_path / f"{args.base_architecture}.json", "w") as f:
            json.dump(results, f, indent=4)
