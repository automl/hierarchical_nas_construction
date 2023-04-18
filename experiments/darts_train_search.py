import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from benchmarks.evaluation.utils import HelperDataset, _load_npy_data
from benchmarks.objectives.custom_nb201.DownsampledImageNet import (
    ImageNet16,
)
from benchmarks.objectives.custom_nb201.evaluate_utils import (
    prepare_seed,
)
from path import Path
from torch.autograd import Variable

import experiments.darts_utils.utils as utils
from experiments.darts_utils.architect import Architect
from experiments.darts_utils.cell_operations import NAS_BENCH_201
from experiments.darts_utils.net2wider import (
    configure_optimizer,
    configure_scheduler,
)
from experiments.darts_utils.search_model import TinyNetwork
from experiments.darts_utils.search_model_gdas import TinyNetworkGDAS

parser = argparse.ArgumentParser("DARTS search on cell-based nb201")
parser.add_argument("--working_directory", type=str, help="where data should be saved")
parser.add_argument(
    "--data_path", type=str, default="datapath", help="location of the data corpus"
)
parser.add_argument(
    "--objective",
    type=str,
    default="cifar10",
    help="choose dataset",
    choices=["cifar10", "cifar100", "ImageNet16-120", "cifarTile", "addNIST"],
)
parser.add_argument(
    "--method",
    type=str,
    default="dirichlet",
    help="choose nas method",
    choices=["darts", "dirichlet", "shapley"],
)
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.001, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=100, help="num of training epochs")
parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument("--cutout_prob", type=float, default=1.0, help="cutout probability")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed", choices=[777, 888, 999]
)
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=3e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding"
)
parser.add_argument(
    "--tau_max",
    type=float,
    default=10,
    help="Max temperature (tau) for the gumbel softmax.",
)
parser.add_argument(
    "--tau_min",
    type=float,
    default=1,
    help="Min temperature (tau) for the gumbel softmax.",
)
parser.add_argument("--k", type=int, default=1, help="partial channel parameter")
#### regularization
parser.add_argument(
    "--reg_type",
    type=str,
    default="l2",
    choices=["l2", "kl"],
    help="regularization type, kl is implemented for dirichlet only",
)
parser.add_argument(
    "--reg_scale",
    type=float,
    default=1e-3,
    help="scaling factor of the regularization term, default value is proper for l2, for kl you might adjust reg_scale to match l2",
)
parser.add_argument(
    "--progressive", action="store_true", default=False, help="use progressive learning"
)
parser.add_argument(
    "--shapley_momentum", type=float, default=0.8, help="Shapley momentum"
)
parser.add_argument(
    "--shapley_step_size", type=float, default=0.1, help="Shapley step size"
)
parser.add_argument(
    "--shapley_samples", type=int, default=10, help="Shapley Monte-Carlo samples"
)
parser.add_argument(
    "--shapley_threshold", type=float, default=0.5, help="Shapley truncation threshold"
)
parser.add_argument("--DEBUG", action="store_true", default=False, help="debug mode")
args = parser.parse_args()

if args.progressive:
    args.save = (
        Path(args.working_directory) / f"{args.method}-progressive" / str(args.seed)
    )
    args.k = 4
else:
    args.save = Path(args.working_directory) / args.method / str(args.seed)
print(args.save)
args.save.makedirs_p()

if args.objective == "cifar10":
    n_classes = 10
elif args.objective == "cifar100":
    n_classes = 100
elif args.objective == "ImageNet16-120":
    n_classes = 120
elif args.objective == "cifarTile":
    n_classes = 4
elif args.objective == "addNIST":
    n_classes = 20


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    prepare_seed(args.seed, workers=3)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.method == "gdas" or args.method == "snas":
        # Create the decrease step for the gumbel softmax temperature
        tau_step = (args.tau_min - args.tau_max) / args.epochs
        tau_epoch = args.tau_max
        if args.method == "gdas":
            model = TinyNetworkGDAS(
                C=args.init_channels,
                N=5,
                max_nodes=4,
                num_classes=n_classes,
                criterion=criterion,
                search_space=NAS_BENCH_201,
            )
        else:
            model = TinyNetwork(
                C=args.init_channels,
                N=5,
                max_nodes=4,
                num_classes=n_classes,
                criterion=criterion,
                search_space=NAS_BENCH_201,
                k=args.k,
                species="gumbel",
            )
    elif args.method == "dirichlet":
        model = TinyNetwork(
            C=args.init_channels,
            N=5,
            max_nodes=4,
            num_classes=n_classes,
            criterion=criterion,
            search_space=NAS_BENCH_201,
            k=args.k,
            species="dirichlet",
            reg_type=args.reg_type,
            reg_scale=args.reg_scale,
        )
    elif args.method == "shapley":
        model = TinyNetwork(
            C=args.init_channels,
            N=5,
            max_nodes=4,
            num_classes=n_classes,
            criterion=criterion,
            search_space=NAS_BENCH_201,
            k=args.k,
            species="shapley",
        )
    elif args.method == "darts":
        model = TinyNetwork(
            C=args.init_channels,
            N=5,
            max_nodes=4,
            num_classes=n_classes,
            criterion=criterion,
            search_space=NAS_BENCH_201,
            k=args.k,
            species="softmax",
        )
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.get_weights(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.objective == "cifar10":
        train_transform, _ = utils._data_transforms_cifar10(args)# pylint: disable=protected-access
        train_data = dset.CIFAR10(
            root=args.data_path, train=True, download=True, transform=train_transform
        )
    elif args.objective == "cifar100":
        train_transform, _ = utils._data_transforms_cifar100(args) # pylint: disable=protected-access
        train_data = dset.CIFAR100(
            root=args.data_path, train=True, download=True, transform=train_transform
        )
    elif args.objective == "ImageNet16-120":
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(
            root=os.path.join(args.data_path, "imagenet16"),
            train=True,
            transform=train_transform,
            use_num_of_class_only=120,
        )
        assert len(train_data) == 151700
    elif args.objective == "cifarTile" or args.objective == "addNIST":
        train_x, train_y, valid_x, valid_y, _, _ = _load_npy_data(
            args.objective, args.data_path
        )
        train_x = np.concatenate((train_x, valid_x), axis=0)
        train_y = np.concatenate((train_y, valid_y), axis=0)
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_data = HelperDataset(train_x, train_y, train_transform)
    else:
        raise NotImplementedError

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
    )

    architect = Architect(model, args)

    genotype = model.genotype()
    genotype_id = genotype.tostr()
    with open(args.save / "genotypes.txt", "w") as f:
        f.write(genotype_id + "\n")

    if args.progressive:
        train_epochs = [50, 50]
        ks = [4, 2]
        num_keeps = [5, 3]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min
        )
    elif "shapley" == args.method:
        train_epochs = [30, 70]  # on DARTS CIFAR-10 [15, 50] -> same portion
        train_epochs = [50, 50]
        accum_shaps = 1e-3 * torch.randn(model.num_edges, model.num_ops).cuda()
        ops = []
        for cell_type in ["nb201cell"]:
            for edge in range(model.num_edges):
                ops.append([f"{cell_type}_{edge}_{i}" for i in range(0, model.num_ops)])
        ops = np.concatenate(ops)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min
        )
    else:
        train_epochs = [args.epochs]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(train_epochs[0]), eta_min=args.learning_rate_min
        )

    if args.DEBUG:
        train_epochs = [0, 50]
        checkpoint = torch.load(
            "/work/dlclarge1/schrodi-hierarchical_nas/neurips_darts_results/nb201_fixed_1_none_new/nb201_cifar10/dirichlet-progressive/777/pretrained.pth"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    for i, current_epochs in enumerate(train_epochs):
        for epoch in range(current_epochs):
            lr = scheduler.get_last_lr()[0]
            logging.info("epoch %d lr %e", epoch, lr)

            if "shapley" == args.method and i == len(train_epochs) - 1:
                shap_nb201cell = shap_estimation(
                    valid_queue,
                    model,
                    ops,
                    num_samples=args.shapley_samples,
                    threshold=args.shapley_threshold,
                )
                accum_shaps = change_alpha(
                    model,
                    shap_nb201cell,
                    accum_shaps,
                    momentum=args.shapley_momentum,
                    step_size=args.shapley_step_size,
                )

            # training
            if args.progressive:
                train_acc, _ = train(
                    train_queue,
                    valid_queue,
                    model,
                    architect,
                    criterion,
                    optimizer,
                    lr,
                    epoch,
                    is_shapley=args.method == "shapley",
                )
            else:
                train_acc, _ = train(
                    train_queue,
                    valid_queue,
                    model,
                    architect,
                    criterion,
                    optimizer,
                    lr,
                    is_shapley=args.method == "shapley",
                )
            print("train_acc %f", train_acc)

            # validation
            valid_acc, _ = infer(valid_queue, model, criterion)
            print("valid_acc %f", valid_acc)

            scheduler.step()
            if args.method == "gdas" or args.method == "snas":
                # Decrease the temperature for the gumbel softmax linearly
                tau_epoch += tau_step
                print("tau %f", tau_epoch)
                model.set_tau(tau_epoch)

            genotype = model.genotype()
            print("genotype = %s", genotype)
            genotype_id = genotype.tostr()
            with open(args.save / "genotypes.txt", "a") as f:
                f.write(genotype_id + "\n")
            model.show_arch_parameters()

        model.train()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": valid_acc,
            },
            args.save / "pretrained.pth"
            if len(train_epochs) > 1 and i == 0
            else "final.pth",
        )

        if args.progressive and (not i == len(train_epochs) - 1):
            model.pruning(num_keeps[i + 1])
            # architect.pruning([model._mask])
            model.wider(ks[i + 1])
            optimizer = configure_optimizer(
                optimizer,
                torch.optim.SGD(
                    model.get_weights(),
                    args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                ),
            )
            scheduler = configure_scheduler(
                scheduler,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(sum(train_epochs)), eta_min=args.learning_rate_min
                ),
            )
            print(f"pruning finished, {num_keeps[i + 1]} ops left per edge")
            print(f"network wider finished, current pc parameter {ks[i + 1]}")

    genotype = model.genotype()
    print("genotype = %s", genotype)
    genotype_id = genotype.tostr()
    with open(args.save / "genotypes.txt", "a") as f:
        f.write(genotype_id + "\n")
    model.show_arch_parameters()


def train(
    train_queue,
    valid_queue,
    model,
    architect,
    criterion,
    optimizer,
    lr,
    epoch: int = None,
    is_shapley: bool = False,
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):  # pylint: disable=redefined-builtin
        model.train()
        n = input.size(0)

        if is_shapley:
            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()
        else:
            input = input.cuda()
            target = target.cuda(non_blocking=True)

        if not is_shapley:
            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

        if is_shapley:
            pass
        elif epoch is not None and epoch >= 10:  # for progressive learning
            architect.step(
                input,
                target,
                input_search,
                target_search,
                lr,
                optimizer,
                unrolled=args.unrolled,
            )
        else:
            architect.step(
                input,
                target,
                input_search,
                target_search,
                lr,
                optimizer,
                unrolled=args.unrolled,
            )

        optimizer.zero_grad()
        if not is_shapley:
            architect.optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if not is_shapley:
            architect.optimizer.zero_grad()

        (prec1,) = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        # top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
        # if "debug" in args.save:
        #     break

    return top1.avg, objs.avg


@torch.no_grad()
def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):  # pylint: disable=redefined-builtin
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        (prec1,) = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        # top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
        # if "debug" in args.save:
        #     break
    return top1.avg, objs.avg


def remove_players(arch_weights, op):
    # selected_cell = str(op.split("_")[0])
    selected_edge_id = int(op.split("_")[1])
    opid = int(op.split("_")[-1])
    # proj_mask = torch.ones_like(normal_weights[selected_eid])
    proj_mask = torch.ones_like(arch_weights[selected_edge_id])
    proj_mask[opid] = 0
    # if selected_cell in ["normal"]:
    #     normal_weights[selected_eid] = normal_weights[selected_eid] * proj_mask
    # else:
    #     reduce_weights[selected_eid] = reduce_weights[selected_eid] * proj_mask
    arch_weights[selected_edge_id] *= proj_mask
    arch_weights[selected_edge_id] /= torch.sum(
        arch_weights[selected_edge_id]
    )  # normalize arch weights
    return arch_weights


def shap_estimation(
    valid_queue, model: nn.Module, players, num_samples: int, threshold: float = 0.5
):
    """
    Implementation of Monte-Carlo sampling of Shapley value for operation importance evaluation
    """
    permutations = None
    n = len(players)
    sv_acc = np.zeros((n, num_samples))

    with torch.no_grad():
        if permutations is None:
            # Keep the same permutations for all batches
            permutations = [np.random.permutation(n) for _ in range(num_samples)]

        for j in range(num_samples):
            x, y = next(iter(valid_queue))
            x, y = x.cuda(), y.cuda(non_blocking=True)
            logits = model(x, arch_weights=None)
            (ori_prec1,) = utils.accuracy(logits, y, topk=(1,))
            acc = ori_prec1.data.item()

            # normal_weights = model.get_projected_weights('normal')
            # reduce_weights = model.get_projected_weights('reduce')
            arch_weights = model.get_projected_weights()

            for i in permutations[j]:
                none_id = NAS_BENCH_201.index("none")
                op_id = int(players[i].split("_")[-1])
                if none_id != op_id:  # R[k] != zero
                    arch_weights = remove_players(arch_weights, players[i])
                    # logits = model(x,  weights_dict={'normal': normal_weights,'reduce':reduce_weights})
                    logits = model(x, arch_weights=arch_weights)
                    (prec1,) = utils.accuracy(logits, y, topk=(1,))
                    new_acc = prec1.item()
                else:
                    new_acc = acc
                sv_acc[i][j] = acc - new_acc
                acc = new_acc
                if acc < threshold * ori_prec1:
                    break
    # result = np.mean(sv_acc, axis=-1) - np.std(sv_acc, axis=-1)
    result = np.mean(sv_acc, axis=-1)
    # shap_acc = np.reshape(result, (2, model.num_edges, model.num_ops))
    # shap_normal, shap_reduce = shap_acc[0], shap_acc[1]
    # return shap_normal, shap_reduce
    shap_acc = np.reshape(result, (model.num_edges, model.num_ops))
    return shap_acc


def change_alpha(model, shap_values, accu_shap_values, momentum=0.8, step_size=0.1):
    assert len(shap_values) == len(model.architecture_parameters)
    shap = [
        torch.from_numpy(shap_values[i]).cuda()
        for i in range(len(model.architecture_parameters))
    ]

    for i, params in enumerate(shap):
        mean = params.data.mean()
        std = params.data.std()
        params.data.add_(-mean).div_(std)

    updated_shap = [
        accu_shap_values[i] * momentum + shap[i] * (1.0 - momentum)
        for i in range(len(model.architecture_parameters))
    ]

    for i, p in enumerate(model._arch_parameters): #pylint: disable=protected-access
        p.data.add_((step_size * updated_shap[i]).to(p.device))

    return updated_shap


if __name__ == "__main__":
    main()
