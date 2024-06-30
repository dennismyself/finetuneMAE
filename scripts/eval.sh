
python main_finetune.py \
    --eval \
    --resume mae_finetuned_vit_base.pth \
    --model vit_base_patch16 \
    --batch_size 16 \
    --data_path "/home/jq271/rds/hpc-work/Dissertation/LLaVA/data/VLMPretrain/mae_data"

    