Polynormer
===============================

[![arXiv](https://img.shields.io/badge/arXiv-2403.01232-b31b1b.svg)](https://arxiv.org/abs/2403.01232)

Polynormer is an expressive graph transformer (GT) that adopts a local-to-global attention scheme with linear complexity. Particularly, the proposed attention module possesses high expressivity in learning equivariant polynomial functions, which map input node features into output node representations. Our experimental results showcase that Polynormer outperforms competitive GNN and GT baselines on a wide range of mainstream datasets, including homophilic graphs, heterophilic graphs, and large graphs with millions of nodes. More details are available in [our paper](https://arxiv.org/abs/2403.01232).

| ![Polynormer.png](/figures/Polynormer.png) | 
|:--:| 
| An overview of the Polynormer architecture |

Citation
------------
If you use Polynormer in your research, please cite our work
published in ICLR'24.

```
@inproceedings{deng2024polynormer,
  title={Polynormer: Polynomial-Expressive Graph Transformer in Linear Time},
  author={Chenhui Deng and Zichao Yue and Zhiru Zhang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=hmv1LpNfXa}
}
```

Requirements
------------
* python 3.9
* pytorch 2.0 (CUDA 11.7)
* torch_geometric 2.3

Python environment setup with Conda (Linux)
------------
```bash
conda create -n polynormer python=3.9
conda activate polynormer
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg

conda clean --all
```

Running Polynormer
------------
```bash
conda activate polynormer

# Running a single experiment on roman-empire
python main.py --dataset roman-empire --hidden_channels 64 --local_epochs 100 --global_epochs 2500 --lr 0.001 --runs 1 --local_layers 10 --global_layers 2 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.15 --num_heads 8 --device 0 --save_model --beta 0.5

# Running all experiments with full batch training
bash run.sh

# Running all experiments with mini-batch training (only required on large graphs)
cd mini-batch
```

Statistics of datasets used in our experiments
-------
### 10 (relatively) small graphs
| Dataset        | Type      | #Nodes  | #Edges  |
| :-----------: |:-------------:| :-------:| :----------:|
| Computer      | Homophily          | 13,752       | 245,861        |
| Photo      | Homophily          | 7,650       | 119,081        |
| CS      | Homophily          | 18,333       | 81,894        |
| Physics      | Homophily          | 34,493       | 247,962        |
| WikiCS      | Homophily          | 11,701       | 216,123        |
| roman-empire      | Heterophily          | 22,662       | 32,927        |
| amazon-ratings      | Heterophily          | 24,492       | 93,050        |
| minesweeper      | Heterophily          | 10,000       | 39,402        |
| tolokers      | Heterophily          | 11,758       | 519,000        |
| questions      | Heterophily          | 48,921       | 153,540        |

### 3 large graphs
| Dataset        | Type      | #Nodes  | #Edges  |
| :-----------: |:-------------:| :-------:| :----------:|
| ogbn-arxiv      | Homophily          | 169, 343       | 1,166,243        |
| ogbn-products      | Homophily          | 2,449,029       | 61,859,140        |
| pokec      | Heterophily          | 1,632,803       | 30,622,564        |

