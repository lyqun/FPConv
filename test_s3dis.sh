gpu=7
model=fpcnn_s3dis
epoch=60
extra_tag=fp_s3dis


nohup python -u tools/test_s3dis.py \
    --gpu ${gpu} \
    --model ${model}\
    --batch_size 24 \
    --weight_dir logs/${extra_tag}/pn2_best_epoch_${epoch}.pth \
    >> test/${extra_tag}_${epoch}.log 2>&1 &
