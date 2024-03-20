#python main-batch.py --dataset pokec --hidden_channels 256 --local_epochs 2000 --global_epochs 0 --batch_size 600000 --lr 0.0005 --runs 1 --local_layers 7 --global_layers 2 --in_drop 0.0 --dropout 0.2 --weight_decay 0.0 --post_bn --eval_step 9 --eval_epoch 1000 --device 0
python main-batch.py --dataset pokec --hidden_channels 256 --local_epochs 2000 --global_epochs 0 --batch_size 400000 --lr 0.0005 --runs 1 --local_layers 7 --global_layers 2 --in_drop 0.0 --dropout 0.2 --weight_decay 0.0 --post_bn --eval_step 9 --eval_epoch 1000 --device 0

