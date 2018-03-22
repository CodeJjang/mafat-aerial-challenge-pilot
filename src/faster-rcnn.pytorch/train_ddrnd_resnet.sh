python trainval_net.py --dataset ddrnd --net res101 \
                       --bs 1 --nw 2 \
                       --lr 4e-3 --lr_decay_step 8 --epochs 11 \
                       --cuda --mGPUs \
                       --save_dir saved_models
                       --checksession=1
                       --checkepoch=8
                       --checkpoint=3173
                       --r=True
