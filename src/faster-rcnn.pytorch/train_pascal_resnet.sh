python trainval_net.py --dataset pascal_voc --net res101 \
                       --bs 8 --nw 2 \
                       --lr 4e-3 --lr_decay_step 8 --epochs 2 \
                       --cuda --mGPUs \
                       --save_dir saved_models
