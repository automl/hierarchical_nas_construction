import logging
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

from benchmarks.search_spaces.darts_cnn import utils
from benchmarks.search_spaces.darts_cnn.model import (
    NetworkCIFAR as Network,
)

TORCH_VERSION = torch.__version__


def train_evaluation(genotype, data, seed, save_path):
    dataset = "cifar10"
    batch_size = 96
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    gpu = 0
    epochs = 600
    init_channels = 36
    layers = 20
    auxiliary = True
    auxiliary_weight = 0.4
    cutout = True
    cutout_length = 16
    drop_path_prob = 0.3
    grad_clip = 5

    CIFAR_CLASSES = 10
    if dataset == "cifar100":
        CIFAR_CLASSES = 100

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(save_path, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(f"Conda environment {os.environ['CONDA_DEFAULT_ENV']}")
    logging.info(f"torch version: {TORCH_VERSION}")
    logging.info(f"batch size: {batch_size}")

    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(seed)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    logging.info("gpu device = %d" % gpu)

    # genotype = eval("genotypes.%s" % args.arch)
    # print("---------Genotype---------")
    logging.info(genotype)
    print("--------------------------")
    model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    if str(TORCH_VERSION) == "1.3.1":
        logging.info("Data Parallel")
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    (
        train_transform,
        valid_transform,
    ) = utils._data_transforms_cifar10(  # pylint: disable=protected-access
        cutout, cutout_length
    )
    if dataset == "cifar100":
        train_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=train_transform
        )
        valid_data = dset.CIFAR100(
            root=data, train=False, download=True, transform=valid_transform
        )
    else:
        train_data = dset.CIFAR10(
            root=data, train=True, download=True, transform=train_transform
        )
        valid_data = dset.CIFAR10(
            root=data, train=False, download=True, transform=valid_transform
        )

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))
    best_acc = 0.0
    valid_accs = []
    for epoch in range(epochs):
        logging.info("epoch %d lr %e", epoch, scheduler.get_lr()[0])
        if str(TORCH_VERSION) == "1.3.1":
            model.module.drop_path_prob = drop_path_prob * epoch / epochs
        else:
            model.drop_path_prob = drop_path_prob * epoch / epochs

        train_acc, _ = train(
            train_queue=train_queue,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=grad_clip,
            auxiliary=auxiliary,
            auxiliary_weight=auxiliary_weight,
            report_freq=report_freq,
        )
        logging.info("train_acc %f", train_acc)

        valid_acc, _ = infer(
            valid_queue=valid_queue,
            model=model,
            criterion=criterion,
            report_freq=report_freq,
        )
        valid_accs.append(float(valid_acc.cpu().detach().numpy()))
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info("valid_acc %f, best_acc %f", valid_acc, best_acc)

        scheduler.step()
        if epoch == epochs - 1:
            utils.save(model, os.path.join(save_path, "weights.pt"))

    del model
    del criterion
    del optimizer
    del scheduler

    logging.getLogger().removeHandler(logging.getLogger().handlers[0])

    return valid_accs


def train(
    train_queue,
    model,
    criterion,
    optimizer,
    grad_clip,
    auxiliary,
    auxiliary_weight,
    report_freq,
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(  # pylint: disable=redefined-builtin
        train_queue
    ):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % report_freq == 0:
            logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(  # pylint: disable=redefined-builtin
            valid_queue
        ):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % report_freq == 0:
                logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg
