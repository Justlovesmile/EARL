## DOTA
python train_net.py --num-gpus 4 --resume \
   --config-file ./configs/DOTAv1/EARL_VAF_R50_600_bs64.yaml

python ./test/json_to_txt_and_merge.py --pred_coco_json ./work_dir/DOTAv1/EARL_VAF_R50_600_bs64/inference/coco_instances_results.json --coco_json /home/sgiit/SGIIT/xmj/DOTA/test600/DOTA1_0_test600.json --dota_version v1

zip -r ./test/EARL_VAF_R50_600_bs64.zip ./output_txt_merge

## DIOR
# python train_net.py --num-gpus 4 --resume \
#     --config-file ./configs/DIOR-R/EARL_VAF_R50_800_bs64.yaml
    
## HRSC2016
# python train_net.py --num-gpus 4 --resume \
#     --config-file ./configs/HRSC2016/EARL_VAF_R50_800_bs16.yaml