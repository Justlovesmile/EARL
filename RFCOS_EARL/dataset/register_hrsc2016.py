import os
from detectron2.data.datasets.register_coco import register_coco_instances


data_root = "HRSC2016/"

_PREDEFINED_SPLITS = {}
_PREDEFINED_SPLITS["hrsc2016"] = {
    "hrsc_trainval": (data_root+'HRSC2016/FullDataSet/AllImages', data_root+"HRSC2016_trainval/HRSC2016_trainval.json"),
    "hrsc_test": (data_root+'HRSC2016/FullDataSet/AllImages', data_root+"HRSC2016_test/HRSC2016_test.json"),
}

HRSC_CATEGORIES = [
    {'id': 1, 'name': 'ship'}
]
    
def _get_hrsc_instances_meta():
    thing_ids = [k["id"] for k in HRSC_CATEGORIES]
    assert len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in HRSC_CATEGORIES]
    ret = {
        "thing_dataset": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_builtin_metadata(dataset_name):
    if dataset_name == "hrsc2016":
        return _get_hrsc_instances_meta()
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
