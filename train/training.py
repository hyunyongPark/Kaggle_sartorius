import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator

from detectron2.data import transforms as T
# utils
from metric_dataset import *
#from augmentation import *
setup_logger()

#import torch.distributed as dist
#dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1) # https://github.com/megvii-model/YOLOF/issues/11

def train_proc():
    dataDir=Path('../data/')
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT='bitmask'
    #register_coco_instances('sartorius_train',{}, f'../coco_anot/annotations_train.json', dataDir)
    #register_coco_instances('sartorius_val',{},f'../coco_anot/annotations_val.json', dataDir)
    register_coco_instances('sartorius_train',{}, f'../coco_anot/5foldcleaned/coco_cell_train_fold5-cleaned.json', dataDir)
    register_coco_instances('sartorius_val',{},f'../coco_anot/5foldcleaned/coco_cell_valid_fold5-cleaned.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    #add_efficientnet_config(cfg)
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x.yaml"))

    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ("sartorius_val",)

    cfg.DATALOADER.NUM_WORKERS = 2

    #cfg.MODEL.BACKBONE
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.yaml")
    #cfg.MODEL.WEIGHTS = "./pre_model/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x-62448b9c.pth"
    #cfg.MODEL.WEIGHTS ="./output_livecell2(CV3008)/model_0036119.pth"
    
    #cfg.MODEL.WEIGHTS ="./output_livecell_norm1.3/model_0033711.pth"
    #cfg.MODEL.WEIGHTS ="./output_livecell(CV3008)/model_final.pth"
    #cfg.MODEL.WEIGHTS ="./output_livecell_cascade_diouloss(CV299)/model_0033711.pth"
    #cfg.MODEL.WEIGHTS ="./output_livecell_cascade_GC제거/model_0040935.pth"
    #cfg.MODEL.WEIGHTS ="./output_transferlearning_scheduler(doc)/model_0007017.pth"
    
    #cfg.MODEL.WEIGHTS ="./output_livecell_cascade_lr476_iter많이/model_0084279.pth"
    cfg.MODEL.WEIGHTS ="./output_transferlearning_LiveCell(에폭많이)_lr0006/model_0004113.pth"
    
    #cfg.MODEL.WEIGHTS ="./output_livecell_mask101_lr476_iter많이/model_0098727.pth"
    #cfg.MODEL.WEIGHTS ="./output_transferlearning_LiveCell(mask101)_origindatalr0006/model_0004113.pth"
    
    
    cfg.SEED = 12
    cfg.MODEL.DEVICE = "cuda:1"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0006
    cfg.SOLVER.MAX_ITER = 15000#1000    
    cfg.SOLVER.STEPS = []

    #cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    #cfg.SOLVER.MAX_ITER = 30000
    #cfg.SOLVER.BASE_LR = 0.01
    #cfg.SOLVER.MOMENTUM = 0.9
    #cfg.SOLVER.NESTEROV = False
    #cfg.SOLVER.WEIGHT_DECAY = 0.0001
    #cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    #cfg.SOLVER.GAMMA = 0.1
    #cfg.SOLVER.STEPS = (20000,)
    #cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    #cfg.SOLVER.WARMUP_ITERS = 1000
    #cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"#"value"
    # Maximum absolute value used for clipping gradients
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0#1.0
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    #cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 1.3


    #cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    # When using pre-trained models in Detectron1 or any MSRA models,
    # std has been absorbed into its conv1 weights, so the std needs to be set 1.
    # Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
    #cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

    
    #cfg.MODEL.PIXEL_MEAN = [128., 128., 128.]
    #cfg.MODEL.PIXEL_STD = [11.58, 11.58, 11.58]
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    #cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    #cfg.INPUT.MIN_SIZE_TEST = 800
    #cfg.INPUT.MAX_SIZE_TEST = 1333
    #choose one of ["horizontal", "vertical", "none"]
    #cfg.INPUT.RANDOM_FLIP = "none"

    # `True` if cropping is used for data augmentation during training
    #cfg.INPUT.CROP.ENABLED = True
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    #cfg.INPUT.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    #cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    #_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    #_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
    # ---------------------
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    #cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "diou"
    #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    #cfg.MODEL.FPN.NORM = "GN"
    
    
    cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    cfg.SOLVER.CHECKPOINT_PERIOD=len(DatasetCatalog.get('sartorius_train')) // cfg.SOLVER.IMS_PER_BATCH

    cfg.OUTPUT_DIR = f"./output_transferlearning_LiveCell(mask101)_origindatalr0006_cleaneddata"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print(f"cfg.SOLVER.CLIP_GRADIENTS.ENABLED : {cfg.SOLVER.CLIP_GRADIENTS.ENABLED}")
    print(f"cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE : {cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE}")
    print(f"cfg.SEED : {cfg.SEED}")
    print(f"cfg.SOLVER.LR_SCHEDULER_NAME : {cfg.SOLVER.LR_SCHEDULER_NAME}")
    print(f"cfg.INPUT.CROP.ENABLED : {cfg.INPUT.CROP.ENABLED}")
    print(f'cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE : {cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE}')
    print(f'cfg.INPUT.RANDOM_FLIP : {cfg.INPUT.RANDOM_FLIP}')
    print(f'cfg.MODEL.FPN.NORM : {cfg.MODEL.FPN.NORM}')
    
    trainer = Trainer(cfg)
    #trainer = AugTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get('sartorius_val')
    
    
    
    
if __name__ == '__main__':
    train_proc()