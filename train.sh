python3 train_main.py --trial_name trial_1.0 \
                      --experiment_name debug \
                      --lr 0.0001445 \
                      --batch_size 32 \
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
                      --model_name resnext100 \
                      --imagenet_pretrain True \
                      --deterministic True \
                      --benchmark True \
                      --precision 16 \
                      --distributed_backend dp \
                      --gpus 4 \
                      --min_epochs 0 \
                      --max_epoch 10 \
                      --weights_save_path /data4/rsna/ckpts \
                      --val_check_interval 0.1 \
                      #--auto_lr_find lr \
                      #--auto_scale_batch_size power \
