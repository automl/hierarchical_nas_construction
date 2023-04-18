import logging
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

import benchmarks.search_spaces.darts_cnn.utils as utils

# from hierarchical_nas_benchmarks.search_spaces.darts_cnn.model_search import Network
from benchmarks.search_spaces.darts_cnn.model import (
    NetworkCIFAR as Network,
)

TORCH_VERSION = torch.__version__


def train_search(genotype, data, seed, save_path):
    dataset = "cifar10"
    epochs = 50
    batch_size = 64
    learning_rate = 0.025  # 0.1
    learning_rate_min = 0.0
    momentum = 0.9
    weight_decay = 3e-4
    report_freq = 50
    gpu = 0
    init_channels = 16  # 36
    layers = 8  # 20
    cutout = False
    cutout_length = 16
    # drop_path_prob = 0.3
    grad_clip = 5
    train_portion = 0.5
    auxiliary = False
    # auxiliary_weight = 0.4

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

    CIFAR_CLASSES = 10
    if dataset == "cifar100":
        CIFAR_CLASSES = 100

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

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(
        init_channels, CIFAR_CLASSES, layers, auxiliary=auxiliary, genotype=genotype
    )
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay
    )

    (
        train_transform,
        _,
    ) = utils._data_transforms_cifar10(  # pylint: disable=protected-access
        cutout, cutout_length
    )
    if dataset == "cifar100":
        train_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=train_transform
        )
    else:
        train_data = dset.CIFAR10(
            root=data, train=True, download=True, transform=train_transform
        )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
    )

    # configure progressive parameter
    # ks = [6, 4]
    # num_keeps = [7, 4]
    # train_epochs = [2, 2] if 'debug' in args.save else [25, 25]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=learning_rate_min
    )

    valid_accs = []
    model.drop_path_prob = 0.0
    for epoch in range(epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info("epoch %d lr %e", epoch, lr)

        # training
        train_acc, _ = train(
            train_queue=train_queue,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip=grad_clip,
            report_freq=report_freq,
        )
        logging.info("train_acc %f", train_acc)

        # validation
        valid_acc, _ = infer(
            valid_queue=valid_queue,
            model=model,
            criterion=criterion,
            report_freq=report_freq,
        )
        valid_accs.append(float(valid_acc.cpu().detach().numpy()))
        logging.info("valid_acc %f", valid_acc)

        scheduler.step()
        # utils.save(model, os.path.join(save_path, 'weights.pt'))

    del model
    del criterion
    del optimizer
    del scheduler

    logging.getLogger().removeHandler(logging.getLogger().handlers[0])

    return valid_accs


def train(train_queue, model, criterion, optimizer, grad_clip, report_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(  # pylint: disable=redefined-builtin
        train_queue
    ):
        model.train()
        n = input.size(0)
        input = input.cuda()  # pylint: disable=redefined-builtin
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        logits, _ = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
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
