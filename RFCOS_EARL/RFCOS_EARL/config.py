from detectron2.config import CfgNode as CN


def add_rtodnet_config(cfg):
    """
    Add config for rtodnet.
    """

    # ---------------------------------------------------------------------------- #
    # RFCOS Head
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.RFCOS = CN()

    # This is the number of foreground classes.
    cfg.MODEL.RFCOS.NUM_CLASSES = 80

    cfg.MODEL.RFCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    cfg.MODEL.RFCOS.NUM_CONVS = 4

    # Prior prob for rare case (i.e. foreground) at the beginning of training.
    # This is used to set the bias for the logits layer of the classifier subnet.
    # This improves training stability in the case of heavy class imbalance.
    cfg.MODEL.RFCOS.PRIOR_PROB = 0.01

    # Inference cls score threshold, only anchors with score > INFERENCE_TH are
    # considered for inference (to improve speed)
    cfg.MODEL.RFCOS.SCORE_THRESH_TEST = 0.05
    # Select topk candidates before NMS
    cfg.MODEL.RFCOS.TOPK_CANDIDATES_TEST = 2000
    cfg.MODEL.RFCOS.NMS_THRESH_TEST = 0.5

    # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
    cfg.MODEL.RFCOS.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    # Loss parameters
    cfg.MODEL.RFCOS.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.RFCOS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.RFCOS.SMOOTH_L1_LOSS_BETA = 0.1
    # Options are: "smooth_l1", "giou"
    cfg.MODEL.RFCOS.BBOX_REG_LOSS_TYPE = "smooth_l1"

    # One of BN, SyncBN, FrozenBN, GN
    # Only supports GN until unshared norm is implemented
    cfg.MODEL.RFCOS.NORM = ""


    cfg.MODEL.RFCOS.L1WEIGHT = 0.01

    cfg.MODEL.RFCOS.USE_SIZE_OF_INTEREST = False

    cfg.MODEL.RFCOS.FPN_STRIDES = [8, 16, 32, 64, 128]

    cfg.MODEL.RFCOS.DATA_AUGMENT = False
    
    cfg.MODEL.RFCOS.RANDOM_ROTATE = False
    
    cfg.MODEL.RFCOS.PIXELATE = False

    cfg.MODEL.RFCOS.TOPK_SAMPLE = 5

    cfg.MODEL.RFCOS.USE_NORMALIZE_REG = False

    cfg.MODEL.RFCOS.ANGLE_RANGE = "a360"
    
    cfg.MODEL.RFCOS.RANGE_RATIO = 0.0