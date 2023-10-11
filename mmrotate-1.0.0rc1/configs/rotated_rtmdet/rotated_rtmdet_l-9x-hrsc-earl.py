_base_ = 'rotated_rtmdet_l-9x-hrsc-earl.py'

model = dict(
    train_cfg=dict(
        assigner=dict(
            type='EARLAssigner',
            range_ratio=2,
            topk=17),
        allowed_border=-1,
        spatial_distance_weight=True,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)
