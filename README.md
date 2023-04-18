# Towards Discovering Neural Architectures from Scratch

**Note that this repository only contains the implementation of the search spaces, evaluation pipelines, and experiment scripts. The main implementation (e.g., how to construct architectures etc.) of our approach is implemented as part of the [NePS](https://github.com/automl/neps) project.**

This repository contains the implementation of our paper "Towards Discovering Neural Architectures from Scratch",
that treats neural architectures as algebraic terms and implements the construction mechanism of algebraic terms with context-free grammars.
For more details, please refer to [our paper](https://arxiv.org/abs/2211.01842).

## 1. Installation
1. Clone this repository.

2. Create a conda environment

```bash
conda create -n hnas python=3.7
```

and activate it

```bash
conda activate hnas
```

3. Install poetry

```bash
bash install_dev_utils/poetry.sh
```

4. Run `poetry install` (this can take quite a while) and then run `pip install opencv-python`.

## 2. Reproducing the paper results
### 2.1 Search on the cell-based or hierarchical NAS-Bench-201 search space
To reproduce those search experiments, run

```bash
python experiments/optimize.py \
--working_directory $working_directory \
--data_path $data_path \
--search_space $search_space \
--objective $objective \
--searcher $searcher \
--surrogate_model $surrogate_model \
--seed $seed \
--pool_strategy evolution \
--pool_size 200 \
--n_init 10 \
--log \
--p_self_crossover 0.5
```
where `$working_directory` and `$data_path` are the directory you want to save to or the path to the dataset, respectively. The other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `search_space`         | `nb201_variable_multi_multi` (hierarchical) or `nb201_fixed_1_none` (cell-based)     |
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |
| `searcher`      | `bayesian_optimization`, `random search`, `regularized_evolution`, or `àssisted_regularized_evolution`     |
| `surrogate_model`       | `gpwl_hierarchical` (hWL), `gpwl` ([WL](https://openreview.net/forum?id=j9Rv7qdXjd)), or `gp_nasbot` ([NASBOT](https://proceedings.neurips.cc/paper/2018/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html)) (only active if `searcher` is set to `bayesian_optimization`) |
| `seed`      | `777`, `888`, `999`                     |

To run DARTS (or improved versions of DARTS) on the cell-based NAS-Bench-201 search space, run
```bash
python darts_train_search.py \
    --working_directory $working_directory \
    --data_path $data_path \
    --objective $objective \
    --seed $seed \
    --method $method
```
where `working_directory` and `data_path` are the directory you want to save to or the path to the dataset, respectively. The other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |
| `seed`      | `777`, `888`, `999`                     |
| `method`      | `darts`, `dirichlet`                     |
Note that add the `--progressive` flag to the above command to run DrNAS with progressive learning scheme.

To evaluate the found architectures, run
```bash
python $WORKDIR/hierarchical_nas_experiments/darts_evaluate.py \
--working_directory $working_directory \
--data_path $data_path \
--objective $objective
```
where `working_directory` and `data_path` are the directory you saved the data to or the path to the dataset, respectively. The other variable can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |

Note that DARTS (or improved versions of it) cannot be applied with further adoption to our hierarchical NAS-Bench-201 search space due to the exponential number of parameters that the supernet would contain.

### 2.2 Search on the activation function search space
To reproduce this search experiment, run

```bash
python experiments/optimize.py \
--working_directory $working_directory \
--data_path $data_path \
--search_space act_cifar10 \
--objective act_cifar10 \
--searcher $searcher \
--surrogate_model $surrogate_model \
--seed $seed \
--pool_strategy evolution \
--pool_size 200 \
--n_init 50 \
--log \
--p_self_crossover 0.5 \
--max_evaluations_total 1000
```
where `$working_directory` and `$data_path` are the directory you want to save to or the path to the dataset, respectively.
The other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `searcher`      | `bayesian_optimization`, `random search`, or `regularized_evolution`     |
| `surrogate_model`       | `gpwl_hierarchical` (hWL), `gpwl` ([WL](https://openreview.net/forum?id=j9Rv7qdXjd)), or `gp_nasbot` ([NASBOT](https://proceedings.neurips.cc/paper/2018/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html)) (only active if `searcher` is set to `bayesian_optimization`) |
| `seed`      | `777`, `888`, `999` (note that we only ran on the seed `777` in our experiments)                    |

### 2.3 Surrogate experiments
**Search has to be run beforehand or data needs to be provided!**

To reproduce our surrogate experiments, run

```bash
python experiments/surrogate_regression.py \
--working_directory $working_directory \
--search_space $search_space \
--objective $objective \
--surrogate_model $surrogate_model \
--n_train $n_train \
--log
```
where `$working_directory` is the directory where the data from the search runs has been saved to and the surrogate results will be saved to. Other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `search_space`         | `nb201_variable_multi_multi` (hierarchical) or `nb201_fixed_1_none` (cell-based)     |
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |
| `surrogate_model`       | `gpwl_hierarchical` (hWL), `gpwl` ([WL](https://openreview.net/forum?id=j9Rv7qdXjd)), or `nasbot` ([NASBOT](https://proceedings.neurips.cc/paper/2018/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html)) (only active if `searcher` is set to `bayesian_optimization`) |
| `n_train`      | `10`, `25`, `50`, `75`, `100`, `150`, `200`, `300`, or `400`                     |

### 2.4 Zero-cost proxy experiments
**Search has to be run beforehand or data needs to be provided!**

To reproduce our zero-cost proxy rank correlation experiments, run

```bash
python experiments/zero_cost_proxy_rank_correlation.py \
--working_directory $working_directory \
--search_space $search_space \
--objective $objective \
--data_path $data_path \
--log
```
where `$working_directory` and `$data_path` are the directory you want to save to and the data from the search runs has been saved to or the path to the dataset, respectively.
Other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `search_space`         | `nb201_variable_multi_multi` (hierarchical) or `nb201_fixed_1_none` (cell-based)     |
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |

### 2.5 NASWOT search experiment
To reproduce the NASWOT search experiment, run
```bash
python experiments/optimize_naswot.py \
--working_directory $working_directory \
--search_space $search_space \
--objective $objective \
--data_path $data_path \
--seed $seed \
--naslib
```
where `$working_directory` and `$data_path` are the directory you want to save to or the path to the dataset, respectively. The other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `search_space`         | `nb201_variable_multi_multi` (hierarchical) or `nb201_fixed_1_none` (cell-based)     |
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |
| `seed`      | `777`, `888`, `999`                     |

### 2.6 DARTS search experiment

## 3. Citing
If you would like to learn more about our work, please read our [paper](https://arxiv.org/abs/2211.01842).
If you find our approach interesting for your own work, please cite the paper:
```
@misc{Schrodi_Towards_Discovering_Neural_2022,
  doi = {10.48550/ARXIV.2211.01842},
  url = {https://arxiv.org/abs/2211.01842},
  author = {Schrodi, Simon and Stoll, Danny and Ru, Binxin and Sukthanker, Rhea and Brox, Thomas and Hutter, Frank},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Towards Discovering Neural Architectures from Scratch},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
## 4. Acknowledgements
We thank the authors of following works for open sourcing their code:
- [NAS-BOWL](https://github.com/xingchenwan/nasbowl): GPWL surrogate model base implementation, NASBOT's graph encoding scheme
- [NASLib](https://github.com/automl/NASLib): base graph class, zero-cost proxies
- [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects): training protocols of NAS-Bench-201 search space
- [CVPR-NAS 2021 Competition Track 3](https://github.com/RobGeada/cvpr-nas-datasets): dataset generation and training protocols for AddNIST and CIFARTile
- [NASWOT](https://github.com/BayesWatch/nas-without-training): implementation of zero-cost proxy search
- [DARTS](https://github.com/quark0/darts), [DrNAS](https://github.com/xiangning-chen/DrNAS): implementation of DARTS training pipeline and DARTS (+ improved versions) search algorithms
