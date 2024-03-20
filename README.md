Polynormer
===============================

[![arXiv](https://img.shields.io/badge/arXiv-2403.01232-b31b1b.svg)](https://arxiv.org/abs/2403.01232)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynormer-polynomial-expressive-graph/node-classification-on-roman-empire)](https://paperswithcode.com/sota/node-classification-on-roman-empire?p=polynormer-polynomial-expressive-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynormer-polynomial-expressive-graph/node-classification-on-amazon-ratings)](https://paperswithcode.com/sota/node-classification-on-amazon-ratings?p=polynormer-polynomial-expressive-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynormer-polynomial-expressive-graph/node-classification-on-minesweeper)](https://paperswithcode.com/sota/node-classification-on-minesweeper?p=polynormer-polynomial-expressive-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynormer-polynomial-expressive-graph/node-classification-on-tolokers)](https://paperswithcode.com/sota/node-classification-on-tolokers?p=polynormer-polynomial-expressive-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/polynormer-polynomial-expressive-graph/node-classification-on-questions)](https://paperswithcode.com/sota/node-classification-on-questions?p=polynormer-polynomial-expressive-graph)

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
* ogb 1.3.6
* gdown

Python environment setup with Conda (Linux)
------------
```bash
conda create -n polynormer python=3.9
conda activate polynormer
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install ogb # only required for ogb graphs
pip install gdown # only required for pokec

conda clean --all
```

Running Polynormer
------------
### Note
We provide the implementation of Polynormer with ReLU. If you would like to see the performance of Polynormer alone, please comment out all ReLU functions.
```bash
conda activate polynormer

# running a single experiment on roman-empire
python main.py --dataset roman-empire --hidden_channels 64 --local_epochs 100 --global_epochs 2500 --lr 0.001 --runs 1 --local_layers 10 --global_layers 2 --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --save_model --beta 0.5 --device 0

# running all experiments with full batch training
bash run.sh

# running all experiments with mini-batch training (only required for ogbn-products and pokec)
cd large_graph_exp
bash products.sh
bash pokec.sh
```

Dataset statistics
-------
| Dataset       | Type      | #Nodes  | #Edges  |
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


Expected results
-------
| Dataset       |  Metric  |   Best baseline from our paper  |  Polynormer-r (first run)  |
| :-----------: |:-------------:| :-------:| :----------:|
| Computer      | Accuracy          | 92.03 (OrderedGNN)       | 94.07        |
| Photo      | Accuracy          | 95.49 (NAGphormer)       | 96.67        |
| CS      | Accuracy          | 95.75 (NAGphormer)       | 95.28        |
| Physics      | Accuracy          | 97.34 (NAGphormer)       | 97.14        |
| WikiCS      | Accuracy          | 79.01 (OrderedGNN)       |  81.20        |
| roman-empire      | Accuracy          | 91.23 (DIR-GNN)       | 92.48        |
| amazon-ratings      | Accuracy          | 53.63 (GraphSAGE)       | 55.04        |
| minesweeper      | ROCAUC          | 93.91 (GAT-sep)       | 97.19        |
| tolokers      | ROCAUC          | 83.78 (GAT-sep)       | 85.15        |
| questions      | ROCAUC          | 78.86 (FSGNN)       | 78.35        |

### Note
We provide the results of Polynormer with ReLU for the first run. Thus, the above results are slightly different from the averaged results over 10 runs in our paper.

Experiments on Large Graphs
-------
```
1. cd large_graph_exp
2. see README.md for instructions
```
