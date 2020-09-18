python3 train_main.py --trial_name trial_1.0 \
                      --experiment_name debug \
                      --lr 0.0001 \
                      --batch_size 128 \
                      --num_workers 16 \
                      --optimizer AdamW \
                      --loss_fn BCE \
                      --class_weight None \
                      --weighted_sampling True \
                      --resize_shape 256 \
                      --crop_shape 224 \
                      --horizontal_flip True \
                      --vertical_flip False \
                      --rotation_range 20 \
                      --normalize False \
                      --model_name densenet121 \
                      --imagenet_pretrain True \
                      --deterministic True \
                      --benchmark True \
                      --precision 16 \
                      --distributed_backend dp \
                      --gpus 4 \
                      --min_epochs 0 \
                      --max_epoch 100 \
                      --weights_save_path /data4/rsna/ckpts \
                      #--auto_lr_find lr \
                      #--auto_scale_batch_size power \