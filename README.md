# Towards Neural Architecture Search from Scratch
This repository contains the implementation of our paper "Towards Neural Architecture Search from Scratch",
a search space design framework expressing neural architectures algebraically that are constructed by context-free grammars.
For more details, please refer to [our paper](TODO).

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
### Search
To reproduce our search experiments, run

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
where you set `$working_directory` and `$data_path` to the directory you want to save to or the path to the dataset, respectively. The other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `search_space`         | `nb201_variable_multi_multi` (hierarchical) or `nb201_fixed_1_none` (cell-based)     |
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |
| `searcher`      | `bayesian_optimization`, `random search`, or `regularized_evolution`     |
| `surrogate_model`       | `gp_hierarchical` (hWL) or `gp` (WL) (only activate if `searcher` is set to `bayesian_optimization`) |
| `seed`      | `777`, `888`, `999`                     |

### Surrogate experiments
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
where you set `$working_directory` is the directory where the data from the search runs have been saved to and the results will be saved to. The other variables can be set as follows:
| variable          | options                                                       |
|--------------------------|-------------------------------------------------------------------|
| `search_space`         | `nb201_variable_multi_multi` (hierarchical) or `nb201_fixed_1_none` (cell-based)     |
| `objective` | `nb201_cifar10`, `nb201_cifar100`, `nb201_ImageNet16-120`, `nb201_cifarTile`, or `nb201_addNIST`   |
| `surrogate_model`       | `gp_hierarchical` (hWL) or `gp` (WL) (only activate if `searcher` is set to `bayesian_optimization`) |
| `n_train`      | `10`, `25`, `50`, `75`, `100`, `150`, `200`, `300`, or `400`                     |

## 3. Citing
If you would like to learn more about our work, please read our [paper](https://arxiv.org/abs/2211.01842).
If you find our approach interesting for your own work, please cite the corresponding paper:
```
@misc{Schrodi_Towards_Neural_Architecture_2022,
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
