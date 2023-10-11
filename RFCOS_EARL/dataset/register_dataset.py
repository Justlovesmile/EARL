import os
from detectron2.data.datasets.register_coco import register_coco_instances

dota_data_root = "DOTA/"

_PREDEFINED_SPLITS = {}
_PREDEFINED_SPLITS["dotav1"] = {
    "v1test600": (dota_data_root+'test600/images', dota_data_root+"test600/DOTA1_0_test600.json"),
    "v1trainval600": (dota_data_root+'trainval600/images', dota_data_root+"trainval600/DOTA1_0_trainval600.json"),
    "v1test600_thr3": (dota_data_root+'test600_thr3/images', dota_data_root+"test600_thr3/DOTA1_0_test600.json"),
    "v1trainval600_thr3": (dota_data_root+'trainval600_thr3/images', dota_data_root+"trainval600_thr3/DOTA1_0_trainval600.json"),
    "v1test600_thr5": (dota_data_root+'test600_thr5/images', dota_data_root+"test600_thr5/DOTA1_0_test600.json"),
    "v1trainval600_thr5": (dota_data_root+'trainval600_thr5/images', dota_data_root+"trainval600_thr5/DOTA1_0_trainval600.json"),
    "v1test800": (dota_data_root+'test800/images', dota_data_root+"test800/DOTA1_0_test800.json"),
    "v1trainval800": (dota_data_root+'trainval800/images', dota_data_root+"trainval800/DOTA1_0_trainval800.json"),
    "v1test1024_512": (dota_data_root+'test1024_512/images', dota_data_root+"test1024_512/DOTA1_0_test1024.json"),
    "v1trainval1024_512": (dota_data_root+'trainval1024_512/images', dota_data_root+"trainval1024_512/DOTA1_0_trainval1024.json"),
    "v1test1024": (dota_data_root+'test1024/images', dota_data_root+"test1024/DOTA1_0_test1024.json"),
    "v1trainval1024": (dota_data_root+'trainval1024/images', dota_data_root+"trainval1024/DOTA1_0_trainval1024.json"),
    "v1test600ms": (dota_data_root+'test600-ms/images', dota_data_root+"test600-ms/DOTA1_0_test600.json"),
    "v1trainval600ms": (dota_data_root+'trainval600-ms/images', dota_data_root+"trainval600-ms/DOTA1_0_trainval600.json"),
}
_PREDEFINED_SPLITS["dotav1.5"] = {
    "v15test600": (dota_data_root+'test600/images', dota_data_root+"test600/DOTA1_0_v15test600.json"),
    "v15test800": (dota_data_root+'test800/images', dota_data_root+"test800/DOTA1_0_v15test800.json"),
    "v15trainval600": (dota_data_root+'v15trainval600/images', dota_data_root+"v15trainval600/DOTA1_0_v15trainval600.json"),
    "v15trainval800": (dota_data_root+'v15trainval800/images', dota_data_root+"v15trainval800/DOTA1_0_v15trainval800.json"),
}


DOTA_CATEGORIES = [
    {'id': 1, 'name': 'plane'},
    {'id': 2, 'name': 'baseball-diamond'},
    {'id': 3, 'name': 'bridge'},
    {'id': 4, 'name': 'ground-track-field'},
    {'id': 5, 'name': 'small-vehicle'},
    {'id': 6, 'name': 'large-vehicle'},
    {'id': 7, 'name': 'ship'},
    {'id': 8, 'name': 'tennis-court'},
    {'id': 9, 'name': 'basketball-court'},
    {'id': 10, 'name': 'storage-tank'},
    {'id': 11, 'name': 'soccer-ball-field'},
    {'id': 12, 'name': 'roundabout'},
    {'id': 13, 'name': 'harbor'},
    {'id': 14, 'name': 'swimming'},
    {'id': 15, 'name': 'helicopter'},
    {'id': 16, 'name': 'container-crane'},
    {'id': 17, 'name': 'airport'},
    {'id': 18, 'name': 'helipad'}
]

DOTA_CATEGORIESv1 = [
    {'id': 1, 'name': 'plane'},
    {'id': 2, 'name': 'baseball-diamond'},
    {'id': 3, 'name': 'bridge'},
    {'id': 4, 'name': 'ground-track-field'},
    {'id': 5, 'name': 'small-vehicle'},
    {'id': 6, 'name': 'large-vehicle'},
    {'id': 7, 'name': 'ship'},
    {'id': 8, 'name': 'tennis-court'},
    {'id': 9, 'name': 'basketball-court'},
    {'id': 10, 'name': 'storage-tank'},
    {'id': 11, 'name': 'soccer-ball-field'},
    {'id': 12, 'name': 'roundabout'},
    {'id': 13, 'name': 'harbor'},
    {'id': 14, 'name': 'swimming-pool'},
    {'id': 15, 'name': 'helicopter'}
]
DOTA_CATEGORIESv15 = [
    {'id': 1, 'name': 'plane'},
    {'id': 2, 'name': 'baseball-diamond'},
    {'id': 3, 'name': 'bridge'},
    {'id': 4, 'name': 'ground-track-field'},
    {'id': 5, 'name': 'small-vehicle'},
    {'id': 6, 'name': 'large-vehicle'},
    {'id': 7, 'name': 'ship'},
    {'id': 8, 'name': 'tennis-court'},
    {'id': 9, 'name': 'basketball-court'},
    {'id': 10, 'name': 'storage-tank'},
    {'id': 11, 'name': 'soccer-ball-field'},
    {'id': 12, 'name': 'roundabout'},
    {'id': 13, 'name': 'harbor'},
    {'id': 14, 'name': 'swimming-pool'},
    {'id': 15, 'name': 'helicopter'},
    {'id': 16, 'name': 'container-crane'}
]

    
def _get_dota_instances_meta():
    thing_ids = [k["id"] for k in DOTA_CATEGORIES]
    assert len(thing_ids) == 18, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DOTA_CATEGORIES]
    ret = {
        "thing_dataset": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_dotav1_instances_meta():
    thing_ids = [k["id"] for k in DOTA_CATEGORIESv1]
    assert len(thing_ids) == 15, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DOTA_CATEGORIESv1]
    ret = {
        "thing_dataset": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_dotav15_instances_meta():
    thing_ids = [k["id"] for k in DOTA_CATEGORIESv15]
    assert len(thing_ids) == 16, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DOTA_CATEGORIESv15]
    ret = {
        "thing_dataset": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret
    
    

def _get_builtin_metadata(dataset_name):
    if dataset_name == "dota":
        return _get_dota_instances_meta()
    elif dataset_name == "dotav1":
        return _get_dotav1_instances_meta()
    elif dataset_name == "dotav1.5":
        return _get_dotav15_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))

def register_all(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

root = '/home/xmj/Datasets/'
register_all(root)
