export MASTER_ADDR=localhost  # or the appropriate master node address
export MASTER_PORT=12355      # or another available port
python main_finetune.py \
    --batch_size 32 \
    --model vit_large_patch16 \
    --finetune /home/jq271/rds/hpc-work/Dissertation/mae/checkpoints/mae_pretrain_vit_large.pth \
    --output_dir "/home/jq271/rds/hpc-work/Dissertation/mae/finetune_checkpoint" \
    --epochs 15 \
    --blr 1e-2 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path "/home/jq271/rds/hpc-work/Dissertation/LLaVA/data/VLMPretrain" \
    --nb_classes 1 \
    --input_size 224 \
    --clip_grad 1 \
    

    