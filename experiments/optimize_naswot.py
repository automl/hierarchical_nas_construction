# This contains implementations of nwot based on the updated version of
# https://github.com/BayesWatch/nas-without-training
# to reflect the second version of the paper https://arxiv.org/abs/2006.04647
# Licensed under MIT license

import argparse
import json
import pickle
import random
import time
from statistics import mean

import numpy as np
import torch
from benchmarks.evaluation.utils import get_train_val_test_loaders
from benchmarks.objectives.hierarchical_nb201 import get_dataloaders
from benchmarks.search_spaces.hierarchical_nb201.graph import (
    NB201_HIERARCHIES_CONSIDERED,
    NB201Spaces,
)
from neps.optimizers.random_search.optimizer import RandomSearch
from neps.search_spaces.search_space import SearchSpace
from path import Path
from tqdm import trange

from experiments.zero_cost_rank_correlation import ZeroCost, evaluate

SearchSpaceMapping = {
    "nb201": NB201Spaces,
}
hierarchies_considered_in_search_space = {**NB201_HIERARCHIES_CONSIDERED}


def hooklogdet(K, labels=None):  # pylint: disable=unused-argument
    s, ld = np.linalg.slogdet(K)  # pylint: disable=unused-variable
    return ld


def random_score(jacob, label=None):  # pylint: disable=unused-argument
    return np.random.normal()


_scores = {"hook_logdet": hooklogdet, "random": random_score}


def get_score_func(score_name):
    return _scores[score_name]


class DropChannel(torch.nn.Module):
    def __init__(self, p, mod):
        super().__init__()
        self.mod = mod
        self.p = p

    def forward(self, s0, s1, droppath):
        ret = self.mod(s0, s1, droppath)
        return ret


class DropConnect(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        channel_size = inputs.shape[1]
        keep_prob = 1 - self.p
        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand(
            [batch_size, channel_size, 1, 1], dtype=inputs.dtype, device=inputs.device
        )
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output


def add_dropout(network, p, prefix=""):
    # p = 0.5
    for attr_str in dir(network):
        target_attr = getattr(network, attr_str)
        if isinstance(target_attr, torch.nn.Conv2d):
            setattr(network, attr_str, torch.nn.Sequential(target_attr, DropConnect(p)))
        # elif isinstance(target_attr, Cell):
        #     setattr(network, attr_str, DropChannel(p, target_attr))
    for n, ch in list(network.named_children()):
        # print(f'{prefix}add_dropout {n}')
        if isinstance(ch, torch.nn.Conv2d):
            setattr(network, n, torch.nn.Sequential(ch, DropConnect(p)))
        # elif isinstance(ch, Cell):
        #     setattr(network, n, DropChannel(p, ch))
        else:
            add_dropout(ch, p, prefix + "\t")


parser = argparse.ArgumentParser(description="NAS Without Training")
parser.add_argument("--working_directory", help="path to data")
parser.add_argument(
    "--search_space",
    default="nb201",
    help="The benchmark dataset to run the experiments.",
    # choices=SearchSpaceMapping.keys(),
)
parser.add_argument(
    "--objective",
    default="nb201_cifar10",
    help="The benchmark dataset to run the experiments.",
)
parser.add_argument("--data_path", help="path to dataset data")
parser.add_argument("--naslib", action="store_true")
parser.add_argument("--resume", action="store_true")

parser.add_argument(
    "--score", default="hook_logdet", type=str, help="the score to evaluate"
)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--kernel", action="store_true")
parser.add_argument("--dropout", action="store_true")
parser.add_argument(
    "--repeat",
    default=1,
    type=int,
    help="how often to repeat a single image with a batch",
)
parser.add_argument(
    "--augtype", default="none", type=str, help="which perturbations to use"
)
parser.add_argument(
    "--sigma", default=0.05, type=float, help='noise level if augtype is "gaussnoise"'
)
parser.add_argument("--GPU", default="0", type=str)
parser.add_argument("--seed", default=777, type=int, choices=[777, 888, 999])
parser.add_argument("--init", default="", type=str)
parser.add_argument("--trainval", action="store_true")
parser.add_argument("--activations", action="store_true")
parser.add_argument("--cosine", action="store_true")
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--n_samples", default=100, type=int)
parser.add_argument("--n_runs", default=10000, type=int)
parser.add_argument(
    "--stem_out_channels",
    default=16,
    type=int,
    help="output channels of stem convolution (nasbench101)",
)
parser.add_argument(
    "--num_stacks", default=3, type=int, help="#stacks of modules (nasbench101)"
)
parser.add_argument(
    "--num_modules_per_stack",
    default=3,
    type=int,
    help="#modules per stack (nasbench101)",
)
parser.add_argument("--num_labels", default=1, type=int, help="#classes (nasbench101)")

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(
    net, x, target, device, args=None
):  # pylint: disable=unused-argument
    net.zero_grad()
    x.requires_grad_(True)
    y, ints = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), ints.detach()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# searchspace = nasspace.get_search_space(args)
idx = args.search_space.find("_")
dataset = args.objective[args.objective.find("_") + 1 :]
search_space = SearchSpaceMapping[args.search_space[:idx]](
    space=args.search_space[idx + 1 :], dataset=dataset, adjust_params=False
)
search_space = SearchSpace(**{"architecture": search_space})

random_architecture_generator = RandomSearch(
    pipeline_space=search_space, initial_design_size=10
)

# train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
if dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
    config, train_loader, ValLoaders = get_dataloaders(
        dataset,
        args.data_path,
        epochs=200,
        gradient_accumulations=1,
        workers=0,
        use_trivial_augment=False,
        eval_mode=False,  # TODO check
    )
    # val_loader["x-valid"]
    if "cifar10" == dataset:
        n_classes = 10
        # test_loader = ValLoaders["ori-test"]
    elif "cifar100" == dataset:
        # test_loader = ValLoaders["ori-test"]
        n_classes = 100
    elif "ImageNet16-120" == dataset:
        # test_loader = ValLoaders["ori-test"]
        n_classes = 120
elif dataset in ["addNIST", "cifarTile"]:
    train_loader, valid_loader, test_loader = get_train_val_test_loaders(
        dataset=dataset,
        data=args.data_path,
        batch_size=64,
        eval_mode=True,
    )
    if "addNIST" == dataset:
        n_classes = 20
    elif "cifarTile" == dataset:
        n_classes = 4
else:
    raise NotImplementedError

args.working_directory = Path(args.working_directory)
args.working_directory.makedirs_p()

times = []
chosen = []
acc = []
val_acc = []
topscores = []
order_fn = np.nanargmax

# if args.dataset == 'cifar10':
#     acc_type = 'ori-test'
#     val_acc_type = 'x-valid'
# else:
#     acc_type = 'x-test'
#     val_acc_type = 'x-valid'

if args.resume:
    raise NotImplementedError

runs = trange(args.n_runs, desc="acc: ")
exponent = 1
best_arch_id = ""
best_arch_idx = -1
scores = []
for N in runs:
    start = time.time()
    # indices = np.random.randint(0,len(searchspace),args.n_samples)

    # npstate = np.random.get_state()
    # ranstate = random.getstate()
    # torchstate = torch.random.get_rng_state()
    # for _ in range(args.n_samples):
    # for arch in indices:
    try:
        # uid = searchspace[arch]
        # network = searchspace.get_network(uid)
        config, config_id, _ = random_architecture_generator.get_config_and_ids()
        network = config["architecture"].to_pytorch()
        network.to(device)

        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != "":
            init_network(network, args.init)  # pylint: disable=undefined-variable

        # random.setstate(ranstate)
        # np.random.set_state(npstate)
        # torch.set_rng_state(torchstate)

        if args.naslib:
            zc_proxy = ZeroCost(
                method_type="nwot", n_classes=n_classes, loss_fn=torch.nn.CrossEntropyLoss
            )
            s = evaluate(zc_proxy=zc_proxy, x_graphs=[network], loader=train_loader)[0]
        else:
            if "hook_" in args.score:
                network.K = np.zeros((args.batch_size, args.batch_size))

                def counting_forward_hook(
                    module, inp, out
                ):  # pylint: disable=unused-argument
                    try:
                        if not module.visited_backwards:
                            return
                        if isinstance(inp, tuple):
                            inp = inp[0]
                        inp = inp.view(inp.size(0), -1)
                        x = (inp > 0).float()
                        K = x @ x.t()
                        K2 = (1.0 - x) @ (1.0 - x.t())
                        network.K = (  # pylint: disable=cell-var-from-loop
                            network.K  # pylint: disable=cell-var-from-loop
                            + K.cpu().numpy()
                            + K2.cpu().numpy()
                        )
                    except Exception:
                        pass

                def counting_backward_hook(
                    module, inp, out  # pylint: disable=unused-argument
                ):
                    module.visited_backwards = True

                for name, module in network.named_modules():
                    if "ReLU" in str(type(module)):
                        # hooks[name] = module.register_forward_hook(counting_hook)
                        module.register_forward_hook(counting_forward_hook)
                        module.register_backward_hook(counting_backward_hook)

            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

            if args.kernel:
                s = get_score_func(args.score)(out, labels)
            elif "hook_" in args.score:
                network(x2.to(device))
                s = get_score_func(args.score)(network.K, target)
            elif args.repeat < args.batch_size:
                s = get_score_func(args.score)(jacobs, labels, args.repeat)
            else:
                s = get_score_func(args.score)(jacobs, labels)

    except Exception as e:
        print(e)
        s = 0.0

    scores.append(s)

    new_best_arch_idx = order_fn(scores)
    if new_best_arch_idx != best_arch_idx:
        best_arch_id = config["architecture"].id
        best_arch_idx = new_best_arch_idx

    # uid = searchspace[best_arch]
    # topscores.append(scores[order_fn(scores)])
    # chosen.append(best_arch)
    # #acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
    # acc.append(searchspace.get_final_accuracy(uid, acc_type, False))

    # if not args.dataset == 'cifar10' or args.trainval:
    #     val_acc.append(searchspace.get_final_accuracy(uid, val_acc_type, args.trainval))
    #    val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time() - start)
    # runs.set_description(f"acc: {mean(acc):.2f}% time:{mean(times):.2f}")
    runs.set_description(f"time:{mean(times):.2f}")

    if N > 0 and (N + 1) % (10 ** exponent) == 0:
        with open(args.working_directory / f"best_{N+1}.json", "w") as fp:
            json.dump(
                {"best_arch_idx": int(best_arch_idx), "best_arch_id": best_arch_id},
                fp,
                indent=2,
            )
        exponent += 1

    if N % 1000:
        npstate = np.random.get_state()
        ranstate = random.getstate()
        torchstate = torch.random.get_rng_state()
        with open(args.working_directory / "checkpoint.pickle", "wb") as fp:
            pickle.dump(
                {
                    "npstate": npstate,
                    "ranstate": ranstate,
                    "torchstate": torchstate,
                },
                fp,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

# print(f"Final mean test accuracy: {np.mean(acc)}")
# if len(val_acc) > 1:
#    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

# state = {'accs': acc,
#          'chosen': chosen,
#          'times': times,
#          'topscores': topscores,
#          }

# dset = args.dataset if not (args.trainval and args.dataset == 'cifar10') else 'cifar10-valid'
# fname = f"{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{dset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
# torch.save(state, fname)
