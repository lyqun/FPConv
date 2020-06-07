gpu=0
model=fpcnn_scannet
epoch=240
extra_tag=fp_scannet


nohup python -u tools/test_scannet.py \
    --gpu ${gpu} \
    --model ${model}\
    --batch_size 24 \
    --with_rgb \
    --with_norm \
    --weight_dir logs/${extra_tag}/pn2_best_epoch_${epoch}.pth \
    >> test/${extra_tag}_${epoch}.log 2>&1 &
