python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --nproc_per_node=4 \
    --master_port=29504 \
    tools/train.py \
    ./configs/rotated_rtmdet/rotated_rtmdet_l-3x-dior-earl.py  \
    --resume \
    --launcher pytorch