gpu=1,5
model=fpcnn_scannet
extra_tag=fp_test

mkdir logs/${extra_tag}

nohup python -u train_scannet.py \
    --model ${model} \
    --batch_size 12 \
    --save_dir logs/${extra_tag} \
    --num_points 8192 \
    --accum 24 \
    --gpu ${gpu} \
    --with_rgb \
    --with_norm \
    >> logs/${extra_tag}/nohup.log 2>&1 &
