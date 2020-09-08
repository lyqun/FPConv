gpu=1,2,3,4
model=fpcnn_s3dis
extra_tag=fp_s3dis

mkdir logs/${extra_tag}

nohup python -u tools/train_s3dis.py \
    --save_dir logs/${extra_tag} \
    --model ${model} \
    --batch_size 8 \
    --gpu ${gpu} \
    >> logs/${extra_tag}/nohup.log 2>&1 &
