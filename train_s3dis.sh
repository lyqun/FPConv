gpu=1
model=fpcnn_s3dis
extra_tag=fp_s3dis

mkdir logs/${extra_tag}

nohup python tools/train_s3dis.py \
    --save_dir logs/${extra_tag} \
    --model ${model} \
    --batch_size 2 \
    --gpu ${gpu} \
    >> logs/${extra_tag}/nohup.log &
