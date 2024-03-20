Large Graph Experiments
===============================

Running Polynormer on large graphs
------------
```bash
conda activate polynormer

# Running experiment on ogbn-arxiv (full batch training)
bash arxiv.sh

# Running experiment on ogbn-products (mini-batch training)
bash products.sh

# Running experiment on pokec (mini-batch training)
bash pokec.sh
```

Dataset statistics
------------
| Dataset        | Type      | #Nodes  | #Edges  |
| :-----------: |:-------------:| :-------:| :----------:|
| ogbn-arxiv      | Homophily          | 169, 343       | 1,166,243        |
| ogbn-products      | Homophily          | 2,449,029       | 61,859,140        |
| pokec      | Heterophily          | 1,632,803       | 30,622,564        |

Expected results
------------
| Dataset        | Metric      | Best baseline from our paper  | Polynormer-r (first run)  |
| :-----------: |:-------------:| :-------:| :----------:|
| ogbn-arxiv      | Accuracy          | 72.41 (GOAT)       | 73.45        |
| ogbn-products      | Accuracy          | 82.00 (GOAT)       | 83.82        |
| pokec      | Accuracy          | 82.04 (LINKX)       | 85.97        |

### Note
We provide the results of Polynormer with ReLU for the first run. Thus, the above results are slightly different from the averaged results over 10 runs in our paper.
