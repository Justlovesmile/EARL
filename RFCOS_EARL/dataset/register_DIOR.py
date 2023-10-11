import os
from detectron2.data.datasets.register_coco import register_coco_instances


data_root = "DIOR/"

_PREDEFINED_SPLITS = {}
_PREDEFINED_SPLITS["DIOR"] = {
    "DIOR_trainval": (data_root+'JPEGImages-trainval', data_root+"DIOR-R_trainval.json"),
    "DIOR_test": (data_root+'JPEGImages-test', data_root+"DIOR-R_test.json"),
}
    
DATA_CATEGORIES = [{'id': 1, 'name': 'windmill'}, 
                   {'id': 2, 'name': 'Expressway-toll-station'}, 
                   {'id': 3, 'name': 'tenniscourt'}, 
                   {'id': 4, 'name': 'vehicle'}, 
                   {'id': 5, 'name': 'baseballfield'}, 
                   {'id': 6, 'name': 'groundtrackfield'}, 
                   {'id': 7, 'name': 'basketballcourt'}, 
                   {'id': 8, 'name': 'ship'}, 
                   {'id': 9, 'name': 'golffield'}, 
                   {'id': 10, 'name': 'dam'}, 
                   {'id': 11, 'name': 'overpass'}, 
                   {'id': 12, 'name': 'Expressway-Service-area'}, 
                   {'id': 13, 'name': 'storagetank'}, 
                   {'id': 14, 'name': 'airplane'}, 
                   {'id': 15, 'name': 'bridge'}, 
                   {'id': 16, 'name': 'airport'}, 
                   {'id': 17, 'name': 'harbor'}, 
                   {'id': 18, 'name': 'stadium'}, 
                   {'id': 19, 'name': 'trainstation'}, 
                   {'id': 20, 'name': 'chimney'}]

def _get_data_instances_meta():
    thing_ids = [k["id"] for k in DATA_CATEGORIES]
    assert len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATA_CATEGORIES]
    ret = {
        "thing_dataset": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_builtin_metadata(dataset_name):
    if dataset_name == "DIOR":
        return _get_data_instances_meta()
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
