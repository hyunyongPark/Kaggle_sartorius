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
from detectron2.structures import polygons_to_bitmask
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
setup_logger()

#import torch.distributed as dist
#dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1) # https://github.com/megvii-model/YOLOF/issues/11

def polygon_to_rle(polygon, shape=(520, 704)):
    #print(polygon)
    mask = polygons_to_bitmask([np.asarray(polygon) + 0.25], shape[0], shape[1])

    rle = mask_util.encode(np.asfortranarray(mask))
    return rle

# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    enc_targs = [polygon_to_rle(enc_targ[0]) for enc_targ in enc_targs]
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)


def train_proc():
    dataDir=Path('../LIVECell_dataset_2021/images')
    cfg = get_cfg()
    register_coco_instances('sartorius_train',{}, '../LIVECell_dataset_2021/livecell_annotations_train.json', dataDir)
    register_coco_instances('sartorius_val',{},'../LIVECell_dataset_2021/livecell_annotations_val.json', dataDir)
    register_coco_instances('sartorius_test',{}, '../LIVECell_dataset_2021/livecell_annotations_test.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')
    
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train", "sartorius_test")
    cfg.DATASETS.TEST = ("sartorius_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SEED = 225
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.000476
    cfg.SOLVER.MAX_ITER = 100000
    cfg.SOLVER.STEPS = []       
    cfg.SOLVER.CHECKPOINT_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    cfg.MODEL.DEVICE = "cuda:1"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    
    #cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    #cfg.SOLVER.MOMENTUM = 0.9
    #cfg.SOLVER.NESTEROV = False
    #cfg.SOLVER.WEIGHT_DECAY = 0.0001
    ##cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    #cfg.SOLVER.GAMMA = 0.1
    #cfg.SOLVER.STEPS = (30000,)
    #cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    #cfg.SOLVER.WARMUP_ITERS = 1000
    #cfg.SOLVER.WARMUP_METHOD = "linear"
    
    
    #cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    
    # If True, augment proposals with ground-truth boxes before sampling proposals to
    # train ROI heads.
    #cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True        
    
    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision
    # detections that will slow down inference post processing steps (like NMS)
    # A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
    # inference.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    
    cfg.TEST.EVAL_PERIOD = (len(DatasetCatalog.get('sartorius_train')) + len(DatasetCatalog.get('sartorius_test'))) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
    
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    #cfg.INPUT.MIN_SIZE_TRAIN = (440, 480, 520, 580, 620)
    #cfg.INPUT.MAX_SIZE_TRAIN = 1333
    #cfg.INPUT.MIN_SIZE_TEST = 800
    #cfg.INPUT.MAX_SIZE_TEST = 1333
    #choose one of ["horizontal", "vertical", "none"]
    #cfg.INPUT.RANDOM_FLIP = "none"
    
    
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"#"value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.
    #cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 1.3
    
    cfg.OUTPUT_DIR = f"./output_livecell_mask101_lr476_iter많이"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(cfg.OUTPUT_DIR)
    print(f"cfg.MODEL.PIXEL_MEAN : {cfg.MODEL.PIXEL_MEAN}")
    print(f"cfg.MODEL.PIXEL_STD : {cfg.MODEL.PIXEL_STD}")
    print(f'MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE : {cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE}')
    print(f"cfg.SOLVER.CLIP_GRADIENTS.ENABLED : {cfg.SOLVER.CLIP_GRADIENTS.ENABLED}")
    print(f"cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE : {cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE}")
    print(f"cfg.SEED : {cfg.SEED}")
    print(f"cfg.SOLVER.LR_SCHEDULER_NAME : {cfg.SOLVER.LR_SCHEDULER_NAME}")
    print(f"cfg.INPUT.CROP.ENABLED : {cfg.INPUT.CROP.ENABLED}")
    print(f'cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE : {cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE}')
    print(f'cfg.INPUT.RANDOM_FLIP : {cfg.INPUT.RANDOM_FLIP}')
    print(f'cfg.MODEL.FPN.NORM : {cfg.MODEL.FPN.NORM}')
    
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    
    
if __name__ == '__main__':
    train_proc()