export MASTER_ADDR=localhost  # or the appropriate master node address
export MASTER_PORT=12355      # or another available port
torchrun --nproc_per_node=1 main_finetune.py \
    --batch_size 256 \
    --model vit_large_patch16 \
    --finetune /home/jq271/rds/hpc-work/Dissertation/mae/saved_checkpoints/pretrain_checkpoint_large_new/checkpoint-20.pth \
    --output_dir "/home/jq271/rds/hpc-work/Dissertation/mae/saved_checkpoints/finetune_checkpoint_large_imagenet_modify_loader_from_scratch" \
    --epochs 100 \
    --blr 1.5e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path "/home/jq271/rds/hpc-work/Dissertation/LLaVA/data/VLMPretrain" \
    --nb_classes 1 \
    --input_size 224 \
    --clip_grad 1 \
    #--resume '/home/jq271/rds/hpc-work/Dissertation/mae/saved_checkpoints/finetune_checkpoint_large_imagenet/checkpoint-4.pth' \
