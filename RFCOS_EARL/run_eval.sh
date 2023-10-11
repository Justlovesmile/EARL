python ./train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50155 --eval-only \
	--config-file  ./configs/DIOR-R/EARL_VAF_R50_800_bs64.yaml \
	MODEL.WEIGHTS ./work_dir/EARL_VAF_R50_800_bs64/model_0019999.pth