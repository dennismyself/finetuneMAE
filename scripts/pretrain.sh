export MASTER_ADDR=localhost  # or the appropriate master node address
export MASTER_PORT=12355      # or another available port
torchrun --nproc_per_node=4 main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_tiny_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --output_dir "/home/jq271/rds/hpc-work/Dissertation/mae/saved_checkpoints/pretrain_checkpoint_tiny_from_scratch" \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path "/home/jq271/rds/hpc-work/Dissertation/LLaVA/data/VLMPretrain" \
    --input_size 224 \
    #--resume /home/jq271/rds/hpc-work/Dissertation/mae/checkpoints/mae_pretrain_vit_base_full.pth \



    

    