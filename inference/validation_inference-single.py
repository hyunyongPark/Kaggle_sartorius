import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
import cv2
import pycocotools.mask as mask_util
import numpy as np
import tqdm

#from sartorius_vis import show_predictions, show_zoomed

##Helper functions
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy().tolist()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)
def score_all():
    scores = []
    for item in val_ds:
        im =  cv2.imread(item['file_name'])
        pred = predictor(im)       
        sc = score(pred, item)
        scores.append(sc)
        
    return np.mean(scores)

def run():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    cfg.MODEL.WEIGHTS = './final_models/tf_live0006_origin00055_model_0000725.pth'  
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda:1"
    predictor = DefaultPredictor(cfg)
    
    register_coco_instances('sartorius_val',{},'../coco_anot/annotations_val.json', 
                            '../data/')

    val_ds = DatasetCatalog.get('sartorius_val')
    
    scores = []
    for item in tqdm.tqdm(val_ds):
        im =  cv2.imread(item['file_name'])
        pred = predictor(im)
        
        targ = item
        if True:
            pred_masks = pred['instances'].pred_masks.cpu().numpy()
            print(pred_masks)
            break
        enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
        enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))
        ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, ious)
            p = tp / (tp + fp + fn)
            prec.append(p)
        #return np.mean(prec)

        sc = np.mean(prec)
        #sc = score(pred, item)
        scores.append(sc)
    print(f"Validation dataset MaP : {np.mean(scores)}")

if __name__ == '__main__':
    run()