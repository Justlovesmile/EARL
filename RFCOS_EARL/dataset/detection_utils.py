# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import cv2
import torch
import numpy as np
from fvcore.transforms.transform import (
    VFlipTransform,
    BlendTransform,
    Transform,
    NoOpTransform
)
from detectron2.structures import (
    BoxMode,
    Instances,
    RotatedBoxes,
)

from detectron2.data import transforms as T
from detectron2.data.transforms import RotationTransform
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.transforms import Augmentation


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    # boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def transform_instance_annotations(
    annotation, transforms, angle_range='a360'
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.txt
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    # bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = annotation["bbox"]
    deg = -np.degrees(bbox[4]) - 90
    # deg = deg if deg != 180 else -deg
    # bbox = bbox[0], bbox[1], bbox[3], bbox[2], deg
    # Note that bbox is 1d (per-instance bounding box)
    if angle_range == 'a360':
        bbox = bbox[0], bbox[1], bbox[3], bbox[2], deg
        annotation["bbox"] = transforms.apply_rotated_box(np.array([bbox]))[0]
    elif angle_range == 'a180':
        bbox[4] = deg
        bbox = transforms.apply_rotated_box(np.array([bbox]))[0]
        # consider using 0-180
        deg = bbox[4]
        if deg > 180:
            deg -= 180
        elif deg < 0:
            deg += 180
            
        deg = deg if deg != 180 else 0
        # deg = deg if deg != 0 else 180  # v4
        
        annotation["bbox"] = bbox[0], bbox[1], bbox[3], bbox[2], deg  #v1
        #annotation["bbox"] = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        # annotation["bbox"] = bbox[0], bbox[1], bbox[2], bbox[3], deg-90 #v2
        # annotation["bbox"] = bbox[0], bbox[1], bbox[3], bbox[2], deg-180 #v3
    elif angle_range == 'a90':
        bbox[4] = deg
        bbox = transforms.apply_rotated_box(np.array([bbox]))[0]
        # consider using 0-90
        deg = bbox[4]
        if deg > 90 and deg <= 180:
            bbox[2], bbox[3], deg = bbox[2], bbox[3], (deg-90)    
        elif deg < 0 and deg >= -90:
            bbox[2], bbox[3], deg = bbox[2], bbox[3], (deg+90)
        elif deg < -90 and deg >= -180:
            bbox[2], bbox[3], deg = bbox[3], bbox[2], (deg+180)
        else:
            bbox[2], bbox[3], deg = bbox[3], bbox[2], deg
            
        # deg = deg if deg != 90 else 0
        
        annotation["bbox"] = bbox[0], bbox[1], bbox[2], bbox[3], deg 

    annotation["bbox_mode"] = BoxMode.XYWHA_ABS

    return annotation

def Rotate_apply_rotated_box(self, rotated_box: np.ndarray) -> np.ndarray:
    #return rotated_box
    box = rotated_box[0]
    # box: cx,cy,long,short,angle between long and y-axis
    # need:
    # box: cx,cy,w,h,a opencv format x-axis counterclock
    deg = box[4]
    if deg > 180:
        deg -= 180  
    elif deg < 0:
        deg += 180
    deg = deg if deg != 180 else 0

    #if deg>90 or deg==90:
    #    box = ((box[0], box[1]), (box[2], box[3]), -deg+90)
    #elif deg<90 or deg==0:
    #    box = ((box[0], box[1]), (box[3], box[2]), -deg)
    box = ((box[0], box[1]), (box[3], box[2]), -deg)
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] = cv2.boxPoints(box).tolist()
    ((x,y),(w,h),a) = cv2.minAreaRect(np.int0(self.apply_coords([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])))
    #print("new_box:",x,y,w,h,a)
    if w>h:
        return [[x,y,max(w,h),min(w,h),-a+90]]
    else:
        return [[x,y,max(w,h),min(w,h),-a]]

def VFlip_apply_image(self, img: np.ndarray) -> np.ndarray:
    """
    Flip the image(s).

    Args:
        img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
            of type uint8 in range [0, 255], or floating point in range
            [0, 1] or [0, 255].
    Returns:
        ndarray: the flipped image(s).
    """
    tensor = torch.from_numpy(np.ascontiguousarray(np.array(img)))
    if len(tensor.shape) == 2:
        # For dimension of HxW.
        tensor = tensor.flip((-2))
    elif len(tensor.shape) > 2:
        # For dimension of HxWxC, NxHxWxC.
        tensor = tensor.flip((-3))
    return tensor.numpy()

def VFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 1] = transform.height - rotated_boxes[:, 1]
    # Transform angle
    ind_min_0 = rotated_boxes[:, 4] < 0
    ind_max_0 = rotated_boxes[:, 4] >= 0
    ind_eq_0 = rotated_boxes[:, 4] == 180
    rotated_boxes[ind_min_0, 4] = - 180 - rotated_boxes[ind_min_0, 4]
    rotated_boxes[ind_max_0, 4] = 180 - rotated_boxes[ind_max_0, 4]
    rotated_boxes[ind_eq_0, 4] = -180
    return rotated_boxes

def Blend_rotated_box(transform, rotated_boxes):
    return rotated_boxes

RotationTransform.register_type("rotated_box", Rotate_apply_rotated_box)
VFlipTransform.register_type("rotated_box", VFlip_rotated_box)
VFlipTransform.register_type("image", VFlip_apply_image)
BlendTransform.register_type("rotated_box", Blend_rotated_box)

class PixelateTransform(Transform):
    def __init__(self, severity=2):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        import imgaug.augmenters as iaa
        seq = iaa.Sequential([
            iaa.imgcorruptlike.Pixelate(severity=self.severity)
        ])
        return seq(image=img)
        
    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation
    
PixelateTransform.register_type("rotated_box", Blend_rotated_box)

class PixelateTrans(Augmentation):
    def __init__(self, severity=1):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        if type(self.severity)==list:
            severity = int(np.random.choice(self.severity))
        else:
            severity = self.severity
        assert severity>=0 and severity<=5 and type(severity)==int, f"Severity {type(severity)}:{severity} is not correct."
        if severity==0:
            return NoOpTransform()
        return PixelateTransform(severity)


def build_augmentation(cfg, is_train, random_rotate=False, pixelate=False):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.MODEL.RFCOS.DATA_AUGMENT:
        # augmentation.append(T.RandomFlip())
        augmentation.append(T.RandomFlip(horizontal=False, vertical=True))
        augmentation.append(T.RandomFlip(horizontal=True, vertical=False))
        if random_rotate:
            augmentation.append(T.RandomRotation(angle=[0, 90, 180, -90],sample_style="choice"))
            augmentation.append(T.RandomRotation(angle=[0, 30, 60, 90],sample_style="choice"))
            #augmentation.append(T.RandomRotation(angle=[60],sample_style="choice"))
        if pixelate:
            augmentation.append(PixelateTrans(severity=[0,1,2]))
        # augmentation.append(T.RandomBrightness(0.7, 1.3))
        # augmentation.append(T.RandomContrast(0.7, 1.3))
        # augmentation.append(T.RandomSaturation(0.7, 1.3))
    return augmentation


build_transform_gen = build_augmentation
