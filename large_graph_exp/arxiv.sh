GPU=1

python main-arxiv.py --dataset ogbn-arxiv --hidden_channels 256 --local_epochs 2000 --global_epochs 0 --lr 0.0005 --runs 1 --local_layers 7 --global_layers 2 --device $GPU --post_bn
