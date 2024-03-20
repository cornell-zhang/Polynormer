python main-batch.py --dataset ogbn-products --local_attn --hidden_channels 32 --num_heads 8 --local_epochs 1000 --global_epochs 0 --lr 0.0005 --batch_size 100000 --runs 1 --local_layers 10 --global_layers 2 --pre_ln --post_bn --in_drop 0.2 --weight_decay 0.0 --eval_step 9 --eval_epoch 500 --device 0

