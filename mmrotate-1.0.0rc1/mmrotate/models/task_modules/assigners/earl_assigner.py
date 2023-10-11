# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmcv.ops import convex_iou, points_in_polygons
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes
from mmdet.structures.bbox import BaseBoxes

@TASK_UTILS.register_module()
class EARLAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox. Each
    proposals will be assigned with `0` or a positive integer indicating the
    ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (int): number of priors selected in each level
    """

    def __init__(
        self,
        topk: int = 15,
        range_ratio: float = 2.0
    ) -> None:
        self.topk = topk
        self.range_ratio = range_ratio

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes # gt boxes  (k,5)
        gt_labels = gt_instances.labels # gt labels (k, )
        num_gt = gt_bboxes.size(0)      # num of gts

        decoded_bboxes = pred_instances.bboxes # pred boxes  (n,5)
        #pred_scores = pred_instances.scores    # pred scores (n,classes)
        priors = pred_instances.priors         # anchor points (n,4)
        num_bboxes = decoded_bboxes.size(0)    # num of predtions

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        prior_center = priors[:, :2] # anchor points location (n,2)

        if isinstance(gt_bboxes, BaseBoxes) or isinstance(gt_bboxes, RotatedBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center) # points in the gt box (n,k)
        else:
            raise Exception("Boxes types error. No such function::find_inside_points")

        valid_mask = is_in_gts.sum(dim=1) > 0 # (n,)

        valid_decoded_bbox = decoded_bboxes[valid_mask] # (v,5)
        num_valid = valid_decoded_bbox.size(0)          

        if num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # EARL
        matched_gt_inds,mathched_gt_centers,valid_mask = self.dynamic_ellipse_adaptive_scale_sampling(
            priors,gt_bboxes,gt_labels,is_in_gts)
        
        # convert to AssignResult format
        # num_valid = decoded_bboxes[valid_mask].size(0)
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()

        assign_result = AssignResult(
            num_gt, 
            assigned_gt_inds, # (n,)
            max_overlaps = None,
            labels=assigned_labels  # (n,)
        )
    
        assign_result.set_extra_property('gt_centers', mathched_gt_centers)
        
        return assign_result
    
    def dynamic_ellipse_adaptive_scale_sampling(self, priors, gt_bboxes, gt_labels, is_in_gts):

        prior_stride = priors[:, 2] # points stride    (n,)
        points_lvl = torch.log2(prior_stride).int() # points level (n,) eg.: 3,4,5
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        valid_mask = is_in_gts.sum(dim=1) > 0

        locations = priors
        locations_with_color = points_lvl

        bboxes = gt_bboxes.tensor
        prior_x = priors[:,0] # (n,)
        prior_y = priors[:,1] # (n,)

        x_shift = prior_x[:,None] - bboxes[:,0][None]  # (n,1) - (1,k) = (n,k)
        y_shift = prior_y[:,None] - bboxes[:,1][None]

        center_dist = torch.sqrt(
            torch.pow(x_shift,2)+torch.pow(y_shift,2)
        ) # (n,k)
        center_dist_reload = torch.zeros_like(center_dist)

        is_in_boxes = torch.zeros_like(center_dist).bool()
        is_selected = torch.zeros_like(center_dist[:,0]).bool()

        w_shift = bboxes[:,2][None] - torch.zeros_like(prior_x[:, None]) # (n,k)
        h_shift = bboxes[:,3][None] - torch.zeros_like(prior_x[:, None])
        obj_size_reg = torch.stack([w_shift, h_shift], dim=-1) / 2 
        obj_size_reg_max, _ = obj_size_reg.max(dim=-1)
        obj_size_reg_min, _ = obj_size_reg.min(dim=-1)  
        obj_ratio = (self.range_ratio - (obj_size_reg_min / obj_size_reg_max)) / self.range_ratio

        is_constrain_in_bbox = is_in_gts

        center_dist = center_dist / obj_size_reg_max
        score_dist_idx = torch.argsort(center_dist, dim=0)
        locations_with_color_sdi = locations_with_color[score_dist_idx]

        idx_for_gt_arg = torch.argsort(obj_size_reg_max[0], dim = -1, descending=True).cpu().int().numpy()
        for idx_for_gt in idx_for_gt_arg:
            is_c_in_each_box = is_constrain_in_bbox[:,idx_for_gt] # (n,)

            w = bboxes[idx_for_gt, 2]
            h = bboxes[idx_for_gt, 3]
            angle = bboxes[idx_for_gt,4] # rad
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            X = x_shift[is_c_in_each_box, idx_for_gt] 
            Y = y_shift[is_c_in_each_box, idx_for_gt] 
            Z = (
                    torch.square(X * cos_a + Y * sin_a) / torch.square(w/2)
                )+(
                    torch.square(Y * cos_a - X * sin_a) / torch.square(h/2)
                )
            thr_ellipse = obj_ratio[is_c_in_each_box,idx_for_gt]

            is_in_ellipse = torch.zeros_like(is_c_in_each_box)
            is_in_ellipse[is_c_in_each_box] = Z < thr_ellipse

            score_argsort = torch.argsort(center_dist[is_in_ellipse,idx_for_gt])
            locations_split_fpn = locations_with_color[is_in_ellipse][score_argsort]
            selected = torch.zeros_like(score_argsort).bool()

            total_assign = self.topk

            last_fpn_index = lvl_min
            for fpn_index in range(lvl_max,lvl_min-1,-1):
                if total_assign > 0:
                    last_fpn_index = fpn_index
                    select_from_fpn_idx = (is_selected[is_in_ellipse][score_argsort]==False)&(locations_split_fpn == fpn_index)
                    select_from_fpn = score_argsort[select_from_fpn_idx]
                    size_select_temp = len(select_from_fpn)
                    selected[select_from_fpn[:min(size_select_temp, total_assign)]] = True
                    
                    select_each_box = torch.zeros_like(is_c_in_each_box).bool()
                    select_each_box[is_in_ellipse] = selected
                    is_selected[select_each_box] = True
                    total_assign -= min(size_select_temp, total_assign)
                else:
                    break

            a_select_from_fpn = locations_with_color_sdi == last_fpn_index
            select_from_fpn_top1 = score_dist_idx[:,idx_for_gt][torch.logical_and(a_select_from_fpn[:,idx_for_gt],is_selected==False)][0]
            is_in_boxes[select_from_fpn_top1, idx_for_gt] = True # (n,k)
            center_dist_reload[select_from_fpn_top1, idx_for_gt] = 1.0

            select_each_box = torch.zeros_like(is_c_in_each_box).bool()
            select_each_box[is_in_ellipse] = selected
            is_in_boxes[select_each_box, idx_for_gt] = True
            center_d_temp = 1.0 - center_dist[select_each_box, idx_for_gt]
            if center_d_temp.size(0) > 0:
                center_dist_reload[select_each_box, idx_for_gt] = center_d_temp / center_d_temp.max() 

        is_in_boxes_max, is_in_boxes_inds = is_in_boxes.max(dim=1) # (n,1)

        valid_mask = is_in_boxes_max!=False

        matched_gt_centers = center_dist_reload.max(dim=-1)[0]

        return is_in_boxes_inds[valid_mask], matched_gt_centers, valid_mask
