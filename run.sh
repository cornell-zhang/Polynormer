GPU=0

## heterophilic datasets
python main.py --dataset roman-empire --hidden_channels 64 --local_epochs 100 --global_epochs 2500 --lr 0.001 --runs 1 --local_layers 10 --global_layers 2 --weight_decay 0.0 --dropout 0.3 --global_dropout 0.5 --in_dropout 0.15 --num_heads 8 --device $GPU --save_model --beta 0.5
python main.py --dataset amazon-ratings --hidden_channels 256 --local_epochs 200 --global_epochs 2500 --lr 0.001 --runs 1 --local_layers 10 --global_layers 1 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 2 --device $GPU --save_model
python main.py --dataset minesweeper --hidden_channels 64 --local_epochs 100 --global_epochs 2000 --lr 0.001 --runs 1 --local_layers 10 --global_layers 3 --weight_decay 0.0 --dropout 0.3 --in_dropout 0.2 --num_heads 8 --metric rocauc --device $GPU --save_model
python main.py --dataset tolokers --hidden_channels 60 --local_epochs 100 --global_epochs 800 --lr 0.001 --runs 1 --local_layers 7 --global_layers 4 --weight_decay 0.0 --dropout 0.5 --in_dropout 0.2 --num_heads 16 --metric rocauc --device $GPU --save_model --beta 0.1
python main.py --dataset questions --hidden_channels 64 --local_epochs 200 --global_epochs 1500 --lr 3e-5 --runs 1 --local_layers 5 --global_layers 3 --weight_decay 0.0 --dropout 0.2 --global_dropout 0.5 --num_heads 8 --metric rocauc --device $GPU --in_dropout 0.15 --save_model --beta 0.4 --pre_ln

## homophilic datasets
python main.py --dataset amazon-computer --hidden_channels 64 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs 1 --local_layers 5 --global_layers 1 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $GPU --save_model
python main.py --dataset amazon-photo --hidden_channels 64 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs 1 --local_layers 7 --global_layers 2 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device $GPU --save_model
python main.py --dataset coauthor-cs --hidden_channels 64 --local_epochs 100 --global_epochs 1500 --lr 0.001 --runs 1 --local_layers 5 --global_layers 2 --weight_decay 5e-4 --dropout 0.3 --in_dropout 0.1 --num_heads 8 --device $GPU --save_model
python main.py --dataset coauthor-physics --hidden_channels 32 --local_epochs 100 --global_epochs 1500 --lr 0.001 --runs 1 --local_layers 5 --global_layers 4 --weight_decay 5e-4 --dropout 0.5 --in_dropout 0.1 --num_heads 8 --device $GPU --save_model
python main.py --dataset wikics --hidden_channels 512 --local_epochs 100 --global_epochs 1000 --lr 0.001 --runs 1 --local_layers 7 --global_layers 2 --weight_decay 0.0 --dropout 0.5 --in_dropout 0.5 --num_heads 1 --device $GPU --save_model

