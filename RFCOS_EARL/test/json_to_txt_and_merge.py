import json
from collections import defaultdict
import shutil
import os
import cv2
# from DOTA_devkit.ResultMerge_multi_process import mergebyrec as mergebypoly
from ResultMerge import mergebypoly
import argparse
import tqdm
dota = 1
def parse_args():
    parser = argparse.ArgumentParser(description='json to xml')
    parser.add_argument('--pred_coco_json', default='')
    parser.add_argument('--coco_json', default='/home/xmj/DOTA/test600/DOTA1_0_test600.json', help='json to xml')
    parser.add_argument('--dota_version', default='v1', help='version of dota dataset')
    parser.add_argument('--add', action='store_true', help='add output')
    args = parser.parse_args()

    return args
    
    
def trans_to_txt(pred_coco_json, coco_json, dota_version, add=False):
    json_file = pred_coco_json
    another_json_file = coco_json    
    # Reading data back
    print("Reading json file: {}".format(json_file))
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Reading data back
    print("Reading another json file: {}".format(another_json_file))
    with open(another_json_file, 'r') as f:
        another_data = json.load(f)

    mapper = defaultdict()
    print("Runninng json image id mapping:")
    for idx in tqdm.tqdm(range(len(another_data['images']))):
        image = another_data['images'][idx]
        mapper[image['id']] = image['file_name']


    file_index = defaultdict(list)
    
    if dota_version == 'v1':
        wordname = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']  # 'container-crane', 'airport', 'helipad'
    elif dota_version == 'v1.5':
        wordname = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    elif dota_version == 'v2':
        wordname = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad']
    else:
        print('Error! No such version of dota dataset')
        return False

    # if the model does not predict some cat
    for cat in wordname:
        if cat not in file_index:
            file_index[cat] = []
            
    print("Runninng json data checking:")
    for ix in tqdm.tqdm(range(len(data))):
        single_data = data[ix]
        image_id = single_data['image_id']
        file_name = mapper[image_id]
        name = file_name.split('.p')[0]
        single_data['file_name'] = name
        file_index[wordname[single_data['category_id'] - 1]].append(single_data)


    list_file_index = list(file_index.keys())

    if not os.path.exists('./output_txt'):
        os.mkdir('./output_txt')
    else:
        if add:
            pass
        else:
            shutil.rmtree('./output_txt')
            os.mkdir('./output_txt')

    print("Runninng results writing:")
    for idx in tqdm.tqdm(range(len(list_file_index))):
        file_txt = list_file_index[idx]
        file = './output_txt/' + '{}.txt'.format(file_txt)

        data = file_index[file_txt]
        if add:
            f = open(file, 'a')
        else:
            f = open(file, "w")

        for data_i in data:
            image_id = data_i['file_name']
            bbox = data_i['bbox']
            score = data_i['score']
            bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), -bbox[4])
            bbox = cv2.boxPoints(bbox).reshape(8)
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            if dota == 1:
                line = f"{image_id} {score:.12f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
            if dota == 2:
                line = f"{image_id} {score:.12f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
            f.write(line + '\n')
        f.close()

    if not os.path.exists('./output_txt_merge'):
        os.mkdir('./output_txt_merge')
    else:
        shutil.rmtree('./output_txt_merge')
        os.mkdir('./output_txt_merge')

    mergebypoly('./output_txt', './output_txt_merge', dota=1)


if __name__ == '__main__':
    args = parse_args()
    pred_coco_json = args.pred_coco_json
    coco_json = args.coco_json
    dota_version = args.dota_version
    add = args.add
    trans_to_txt(pred_coco_json, coco_json, dota_version, add)