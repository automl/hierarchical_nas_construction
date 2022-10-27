#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################

import math
import random
from bisect import bisect_right

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


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


class _LRScheduler:
    def __init__(self, optimizer, warmup_epochs, epochs):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.max_epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.current_iter = 0

    @staticmethod
    def extra_repr():
        return ""

    def __repr__(self):
        return "{name}(warmup={warmup_epochs}, max-epoch={max_epochs}, current::epoch={current_epoch}, iter={current_iter:.2f}".format(
            name=self.__class__.__name__, **self.__dict__
        ) + ", {:})".format(
            self.extra_repr()
        )

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def get_min_info(self):
        lrs = self.get_lr()
        return "#LR=[{:.6f}~{:.6f}] epoch={:03d}, iter={:4.2f}#".format(
            min(lrs), max(lrs), self.current_epoch, self.current_iter
        )

    def get_min_lr(self):
        return min(self.get_lr())

    def update(self, cur_epoch, cur_iter):
        if cur_epoch is not None:
            assert (
                isinstance(cur_epoch, int) and cur_epoch >= 0
            ), f"invalid cur-epoch : {cur_epoch}"
            self.current_epoch = cur_epoch
        if cur_iter is not None:
            assert (
                isinstance(cur_iter, float) and cur_iter >= 0
            ), f"invalid cur-iter : {cur_iter}"
            self.current_iter = cur_iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, warmup_epochs, epochs)

    def extra_repr(self):
        return "type={:}, T-max={:}, eta-min={:}".format(
            "cosine", self.T_max, self.eta_min
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if (
                self.current_epoch >= self.warmup_epochs
                and self.current_epoch < self.max_epochs
            ):
                last_epoch = self.current_epoch - self.warmup_epochs
                # if last_epoch < self.T_max:
                # if last_epoch < self.max_epochs:
                lr = (
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * last_epoch / self.T_max))
                    / 2
                )
                # else:
                #  lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_max-1.0) / self.T_max)) / 2
            elif self.current_epoch >= self.max_epochs:
                lr = self.eta_min
            else:
                lr = (
                    self.current_epoch / self.warmup_epochs
                    + self.current_iter / self.warmup_epochs
                ) * base_lr
            lrs.append(lr)
        return lrs


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, milestones, gammas):
        assert len(milestones) == len(gammas), "invalid {:} vs {:}".format(
            len(milestones), len(gammas)
        )
        self.milestones = milestones
        self.gammas = gammas
        super().__init__(optimizer, warmup_epochs, epochs)

    def extra_repr(self):
        return "type={:}, milestones={:}, gammas={:}, base-lrs={:}".format(
            "multistep", self.milestones, self.gammas, self.base_lrs
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.current_epoch >= self.warmup_epochs:
                last_epoch = self.current_epoch - self.warmup_epochs
                idx = bisect_right(self.milestones, last_epoch)
                lr = base_lr
                for x in self.gammas[:idx]:
                    lr *= x
            else:
                lr = (
                    self.current_epoch / self.warmup_epochs
                    + self.current_iter / self.warmup_epochs
                ) * base_lr
            lrs.append(lr)
        return lrs


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, gamma):
        self.gamma = gamma
        super().__init__(optimizer, warmup_epochs, epochs)

    def extra_repr(self):
        return "type={:}, gamma={:}, base-lrs={:}".format(
            "exponential", self.gamma, self.base_lrs
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.current_epoch >= self.warmup_epochs:
                last_epoch = self.current_epoch - self.warmup_epochs
                assert last_epoch >= 0, f"invalid last_epoch : {last_epoch}"
                lr = base_lr * (self.gamma ** last_epoch)
            else:
                lr = (
                    self.current_epoch / self.warmup_epochs
                    + self.current_iter / self.warmup_epochs
                ) * base_lr
            lrs.append(lr)
        return lrs


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, max_LR, min_LR):
        self.max_LR = max_LR
        self.min_LR = min_LR
        super().__init__(optimizer, warmup_epochs, epochs)

    def extra_repr(self):
        return "type={:}, max_LR={:}, min_LR={:}, base-lrs={:}".format(
            "LinearLR", self.max_LR, self.min_LR, self.base_lrs
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.current_epoch >= self.warmup_epochs:
                last_epoch = self.current_epoch - self.warmup_epochs
                assert last_epoch >= 0, f"invalid last_epoch : {last_epoch}"
                ratio = (
                    (self.max_LR - self.min_LR)
                    * last_epoch
                    / self.max_epochs
                    / self.max_LR
                )
                lr = base_lr * (1 - ratio)
            else:
                lr = (
                    self.current_epoch / self.warmup_epochs
                    + self.current_iter / self.warmup_epochs
                ) * base_lr
            lrs.append(lr)
        return lrs


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_optim_scheduler(parameters, config):
    assert (
        hasattr(config, "optim")
        and hasattr(config, "scheduler")
        and hasattr(config, "criterion")
    ), f"config must have optim / scheduler / criterion keys instead of {config}"
    if config.optim == "SGD":
        optim = torch.optim.SGD(
            parameters,
            config.LR,
            momentum=config.momentum,
            weight_decay=config.decay,
            nesterov=config.nesterov,
        )
    elif config.optim == "RMSprop":
        optim = torch.optim.RMSprop(
            parameters, config.LR, momentum=config.momentum, weight_decay=config.decay
        )
    else:
        raise ValueError(f"invalid optim : {config.optim}")

    if config.scheduler == "cos":
        T_max = getattr(config, "T_max", config.epochs)
        scheduler = CosineAnnealingLR(
            optim, config.warmup, config.epochs, T_max, config.eta_min
        )
    elif config.scheduler == "multistep":
        scheduler = MultiStepLR(
            optim, config.warmup, config.epochs, config.milestones, config.gammas
        )
    elif config.scheduler == "exponential":
        scheduler = ExponentialLR(optim, config.warmup, config.epochs, config.gamma)
    elif config.scheduler == "linear":
        scheduler = LinearLR(
            optim, config.warmup, config.epochs, config.LR, config.LR_min
        )
    else:
        raise ValueError(f"invalid scheduler : {config.scheduler}")

    if config.criterion == "Softmax":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.criterion == "SmoothSoftmax":
        criterion = CrossEntropyLabelSmooth(config.class_num, config.label_smooth)
    else:
        raise ValueError(f"invalid criterion : {config.criterion}")
    return optim, scheduler, criterion


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{name}(val={val}, avg={avg}, count={count})".format(
            name=self.__class__.__name__, **self.__dict__
        )


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
