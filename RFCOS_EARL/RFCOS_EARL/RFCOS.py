# -*- coding: utf-8 -*-
import cv2
import math
import torch
import logging
import numpy as np

from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Dict, List, Tuple
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms_rotated, cat, get_norm, nonzero_tuple
from detectron2.structures import (
    ImageList, 
    Instances, 
    RotatedBoxes,
)

from detectron2.utils.events import get_event_storage
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling import META_ARCH_REGISTRY


__all__ = ["RFCOS_EARL"]
INF = 100000000

#进行图片的复制拼接
def to0_1(tensor):
    return (tensor - tensor.min())/(tensor.max() - tensor.min())

def concat_feature_pic(features, features_level=0, data_id=0, boarder=1, num_fea=16):
    image_files = []
    COL = num_fea
    ROW = num_fea
    for index in range(COL * ROW):
        tensor = features[features_level][data_id][index].cpu().detach()
        n_tensor = np.int32(to0_1(tensor).numpy() * 255)
        image_files.append(Image.fromarray(np.uint8(n_tensor))) #读取所有用于拼接的图片
    UNIT_WIDTH_SIZE = features[features_level].size(2)
    UNIT_HEIGHT_SIZE = features[features_level].size(3)
    target = Image.new('P', ((UNIT_WIDTH_SIZE+boarder) * COL, (UNIT_HEIGHT_SIZE+boarder) * ROW), 255) #创建成品图的画布
    #第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    for row in range(ROW):
        for col in range(COL):
            #对图片进行逐行拼接
            target.paste(image_files[COL*row+col], (0 + (UNIT_WIDTH_SIZE+boarder)*col, 0 + (UNIT_HEIGHT_SIZE+boarder)*row))
            
    return target.resize((800, 800))


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def compute_location(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1) + stride // 2
    shift_y = shift_y.reshape(-1) + stride // 2
    shift_zeros = torch.zeros(shift_x.size(0), dtype=shift_x.dtype, device=shift_x.device)
    locations = torch.stack((
        shift_x, shift_y, torch.zeros_like(shift_zeros), torch.zeros_like(shift_zeros), torch.zeros_like(shift_zeros)
    ), dim=1) 
    return locations


def weight_smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, beta: float, reduction: str = "none"
) -> torch.Tensor:

    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    loss = loss.sum(dim=-1) * weight

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@META_ARCH_REGISTRY.register()
class RFCOS_EARL(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone,
        head,
        head_in_features,
        num_classes,
        fpn_strides,
        l1_weight,
        topk_sample = 5,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.1,
        box_reg_loss_type="smooth_l1",
        use_normalize_reg=False,
        range_ratio = 0.0,
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_in_features = head_in_features
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.l1_weight = l1_weight
        self.topk_sample = topk_sample
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.use_normalize_reg = use_normalize_reg
        self.range_ratio = range_ratio if range_ratio!=0.0 else False
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9
        self.max_coor = 0.0


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RFCOS.IN_FEATURES]
        head_in_features = cfg.MODEL.RFCOS.IN_FEATURES
        head = RFCOSHead(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.RFCOS.NUM_CLASSES,
            "head_in_features": head_in_features,
            "fpn_strides": cfg.MODEL.RFCOS.FPN_STRIDES,
            "l1_weight": cfg.MODEL.RFCOS.L1WEIGHT,
            "topk_sample": cfg.MODEL.RFCOS.TOPK_SAMPLE,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.RFCOS.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.RFCOS.FOCAL_LOSS_GAMMA,
            "smooth_l1_beta": cfg.MODEL.RFCOS.SMOOTH_L1_LOSS_BETA,
            "box_reg_loss_type": cfg.MODEL.RFCOS.BBOX_REG_LOSS_TYPE,
            "use_normalize_reg": cfg.MODEL.RFCOS.USE_NORMALIZE_REG,
            "range_ratio": cfg.MODEL.RFCOS.RANGE_RATIO,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.RFCOS.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.RFCOS.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.RFCOS.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results, gt_labels, gt_boxes, locations):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        locations_clone = []
        for ind, l in enumerate(locations):
            locations_clone.append(locations[ind][:, 4].clone() + ind)

        gt_labels_per_img, gt_boxes_per_img = gt_labels[image_index], gt_boxes[image_index]
        locations_cat = torch.cat(locations, dim=0)
        locations_clone_cat = torch.cat(locations_clone, dim=0)
        pos_mask = gt_labels_per_img != self.num_classes
        locations_cat_select = locations_cat[pos_mask]
        locations_clone_select = locations_clone_cat[pos_mask]

        red_color = (0, 100, 255) # BGR
        green_color = (0, 255, 0) # BGR
        blue_color = (255, 0, 0) # BGR
        blue_green_color = (255, 255, 0) # BGR
        blue_red_color = (255, 0, 255) # BGR
        color_index = [red_color, green_color, blue_color, blue_green_color, blue_red_color]
        thickness = 2 # 可以为 0 、4、8

        point_size = 3
        points_list = locations_cat_select[:, 0:2].cpu().int().numpy()
        point_color_list = locations_clone_select.cpu().int().numpy()

        im = cv2.cvtColor(anno_img, cv2.COLOR_RGB2BGR)  
        for point, color in zip(points_list, point_color_list):
            if color >= 0:
                cv2.circle(im, tuple(point), point_size, color_index[color], thickness)
        anno_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)

        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        locations = self.compute_locations(features)
        
        pred_logits, pred_anchor_deltas = self.head(features) # pre_anchor_deltas: (tx,ty,w,h,\theta)
                    
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 5) for x in pred_anchor_deltas]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            gt_labels, gt_boxes, gt_centers = self.label_assignment_earl(locations, gt_instances)
            losses = self.losses(locations, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, gt_centers)
                
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        locations, pred_logits, pred_anchor_deltas, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results, gt_labels, gt_boxes, locations)

            return losses
        else:
            results = self.inference(locations, pred_logits, pred_anchor_deltas, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, gt_centers=None):
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)
        gt_anchor_deltas = torch.stack(gt_boxes)  # (N, R, 4)
        valid_mask = gt_labels >= 0
        pos_mask = gt_labels != self.num_classes
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)
        
        loss_dict = {}

        # classification and regression loss

        if self.box_reg_loss_type == "smooth_l1":
            gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1]
            loss_cls = sigmoid_focal_loss_jit(
                cat(pred_logits, dim=1)[valid_mask],
                gt_labels_target.to(pred_logits[0].dtype),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            loss_dict['loss_cls'] = loss_cls / self.loss_normalizer

            loss_box_reg = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
            loss_dict['loss_box_reg'] = loss_box_reg / self.loss_normalizer * self.l1_weight
        elif self.box_reg_loss_type == "smooth_l1_weight":
            gt_anchor_centers = torch.stack(gt_centers)  # (N, R, 4)

            gt_labels_target = gt_anchor_centers[valid_mask][:, None] * F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[:, :-1]
            loss_cls = sigmoid_focal_loss_jit(
                cat(pred_logits, dim=1)[valid_mask],
                gt_labels_target.to(pred_logits[0].dtype),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )

            loss_dict['loss_cls'] = loss_cls / self.loss_normalizer

            loss_box_reg = weight_smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[pos_mask],
                gt_anchor_deltas[pos_mask],
                gt_anchor_centers[pos_mask],
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
            loss_dict['loss_box_reg'] = loss_box_reg / self.loss_normalizer * self.l1_weight
            #gt_anchor_centers
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        return loss_dict
    
    @torch.no_grad()
    def label_assignment_earl(self, anchors, gt_instances, pred_logits=None, pred_boxes=None, thr_gaussion=1e-5):
        range_ratio = self.range_ratio
        locations = torch.cat(anchors, dim=0)
        locations_with_color = torch.cat([anchors[ind][:, 4] + ind for ind in range(len(anchors))])

        # anchor's x & y of center point
        xs, ys = locations[:, 0], locations[:, 1]

        gt_labels = []
        matched_gt_boxes = []
        gt_centers = []
        for im_i in range(len(gt_instances)):
            # gt instances in im_i'th image
            gt_per_image = gt_instances[im_i]
            bboxes = gt_per_image.gt_boxes.tensor
            labels_per_im = gt_per_image.gt_classes

            # offset between each gt x,y and anchor x,y
            center_x_shift = xs[:, None] - bboxes[:, 0][None]
            center_y_shift = ys[:, None] - bboxes[:, 1][None]

            # distance = offset_x^2+offset_y^2
            center_d = torch.sqrt(torch.pow(center_x_shift, 2) + torch.pow(center_y_shift, 2))
            center_d_reload = torch.zeros_like(center_d)
            if len(gt_per_image) > 0:
                is_in_boxes = torch.zeros_like(center_d).bool()
                is_selected = torch.zeros_like(center_d[:,0]).bool()

                dumy_xy = torch.zeros_like(xs[:, None])
                # w and h from anchor
                w_recover = bboxes[:, 2][None] - dumy_xy
                h_recover = bboxes[:, 3][None] - dumy_xy
                # w/2 and h/2
                obj_size_reg = torch.stack([w_recover, h_recover], dim=-1) / 2
                obj_size_reg_max, _ = obj_size_reg.max(dim=-1)
                obj_size_reg_min, _ = obj_size_reg.min(dim=-1)  
                # retio = 1 - \frac{min}{2*max}
                obj_ratio = (2 - (obj_size_reg_min / obj_size_reg_max))/2
                # distance < (max length / 2) 
                is_constrain_in_box = center_d < obj_size_reg_max*2 # 在大圆内
                
                center_d = center_d / obj_size_reg_max
                
                scores_d_inds = torch.argsort(center_d, dim=0)

                locations_with_color_sdi = locations_with_color[scores_d_inds]

                index_for_gt_arg =  torch.argsort(obj_size_reg_max[0],dim=-1,descending=True).cpu().int().numpy()

                for index_for_gt in index_for_gt_arg:
                    is_c_in_each_box = is_constrain_in_box[:, index_for_gt]

                    w= bboxes[index_for_gt, 2] 
                    h= bboxes[index_for_gt, 3] 
                    angle = torch.deg2rad(180 - bboxes[index_for_gt, 4])
                    cos_a = torch.cos(angle)
                    sin_a = torch.sin(angle)
                    X = center_x_shift[is_c_in_each_box, index_for_gt] 
                    Y = center_y_shift[is_c_in_each_box, index_for_gt] 
                    Z = (
                            torch.square(X * cos_a + Y * sin_a) / torch.square(w/2)
                        )+(
                            torch.square(Y * cos_a - X * sin_a) / torch.square(h/2)
                        )
                    if range_ratio:
                        thr_gaussion = range_ratio
                    else:
                        thr_gaussion = obj_ratio[is_c_in_each_box, index_for_gt]

                    is_constrain_in_rbox = torch.zeros_like(is_c_in_each_box)
                    is_constrain_in_rbox[is_c_in_each_box] = Z < thr_gaussion

                    score_argsort = torch.argsort(center_d[is_constrain_in_rbox, index_for_gt])
                    locations_split_fpn = locations_with_color[is_constrain_in_rbox][score_argsort]
                    selected = torch.zeros_like(score_argsort).bool()
                    
                    tollal_assign = self.topk_sample

                    last_fpn_index = 0
                    for fpn_index in range(4,-1,-1):
                        if tollal_assign > 0:
                            last_fpn_index = fpn_index
                            
                            select_from_fpn_idx = (is_selected[is_constrain_in_rbox][score_argsort]==False)&(locations_split_fpn == fpn_index)
                            select_from_fpn = score_argsort[select_from_fpn_idx]
                            
                            size_select_temp = len(select_from_fpn)
                            selected[select_from_fpn[:min(size_select_temp, tollal_assign)]] = True
                            
                            select_each_box = torch.zeros_like(is_c_in_each_box).bool()
                            select_each_box[is_constrain_in_rbox] = selected
                            is_selected[select_each_box] = True
                            
                            tollal_assign -= min(size_select_temp, tollal_assign)
                        else:
                            break

                    a_select_from_fpn = locations_with_color_sdi == last_fpn_index
                    select_from_fpn_top1 = scores_d_inds[:,index_for_gt][torch.logical_and(a_select_from_fpn[:,index_for_gt],is_selected==False)][0]
                    is_in_boxes[select_from_fpn_top1, index_for_gt] = True
                    center_d_reload[select_from_fpn_top1, index_for_gt] = 1.0
        
                    select_each_box = torch.zeros_like(is_c_in_each_box).bool()
                    select_each_box[is_constrain_in_rbox] = selected
                    is_in_boxes[select_each_box, index_for_gt] = True
                    center_d_temp = 1.0 - center_d[select_each_box, index_for_gt]
                    if center_d_temp.size(0) > 0:
                        center_d_reload[select_each_box, index_for_gt] = center_d_temp / center_d_temp.max() 

                is_in_boxes_max, is_in_boxes_inds = is_in_boxes.max(dim=1)
                matched_gt_boxes_i = bboxes[is_in_boxes_inds]
                matched_gt_boxes_i = matched_gt_boxes_i - locations

                if self.use_normalize_reg:
                    matched_gt_boxes_i[:, 2:4] = matched_gt_boxes_i[:, 2:4] / 5

                gt_labels_i = labels_per_im[is_in_boxes_inds]
                gt_labels_i[is_in_boxes_max == False] = self.num_classes
                gt_centers_i = center_d_reload.max(dim=-1)[0]
            else:
                gt_labels_i = torch.zeros_like(xs) + self.num_classes
                matched_gt_boxes_i = torch.zeros_like(locations)
                gt_centers_i = torch.zeros_like(xs) 

            gt_labels.append(gt_labels_i.to(torch.int64))
            matched_gt_boxes.append(matched_gt_boxes_i)
            gt_centers.append(gt_centers_i)

        return gt_labels, matched_gt_boxes, gt_centers
    
    def inference(
        self,
        anchors: List[Tensor],
        pred_logits: List[Tensor],
        pred_anchor_deltas: List[Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        results: List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
        self,
        anchors: List[Tensor],
        box_cls: List[Tensor],
        box_delta: List[Tensor],
        image_size: Tuple[int, int],
    ):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        for l in range(len(box_cls)):
            box_cls_i, box_reg_i, anchors_i = box_cls[l], box_delta[l], anchors[l]
            predicted_prob = box_cls_i.flatten().sigmoid_()

            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='trunc')
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            predicted_boxes = box_reg_i + anchors_i

            if self.use_normalize_reg:
                predicted_boxes[:, 2:4] = predicted_boxes[:, 2:4] * self.fpn_strides[l]

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        
        keep = batched_nms_rotated(boxes_all, scores_all, class_idxs_all, self.test_nms_thresh)

        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = RotatedBoxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_location(
                h, w, self.fpn_strides[level],
                feature.device
            )
            if self.max_coor < locations_per_level.max():
                self.max_coor = locations_per_level.max()
            locations.append(locations_per_level)
        return locations

class RFCOSHead(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        num_classes,
        conv_dims: List[int],
        norm="",
        prior_prob=0.01,
    ):
        super().__init__()

        if norm == "BN" or norm == "SyncBN":
            logger = logging.getLogger(__name__)
            logger.warn("Shared norm does not work well for BN, SyncBN, expect poor results")

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip([input_shape[0].channels] + conv_dims, conv_dims):
            cls_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            conv_dims[-1], num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dims[-1], 5, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    # torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):

        return {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.RFCOS.NUM_CLASSES,
            "conv_dims": [input_shape[0].channels] * cfg.MODEL.RFCOS.NUM_CONVS,
            "prior_prob": cfg.MODEL.RFCOS.PRIOR_PROB,
            "norm": cfg.MODEL.RFCOS.NORM,
        }

    def forward(self, features: List[Tensor]):
        logits = []
        bbox_reg = []
        for i, feature in enumerate(features):
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg