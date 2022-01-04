import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import torch
import detectron2
from tqdm.auto import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.modeling.test_time_augmentation import *
import pycocotools.mask as mask_util
from PIL import Image
import matplotlib.pyplot as plt
from fastcore.all import *
from ensemble_boxes import *
import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('GPU is available')
else:
    DEVICE = torch.device('cpu')
    print('CPU is used')
print('detectron ver:', detectron2.__version__)


##Helper functions
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    #pred_masks = pred['instances'].pred_masks.cpu().numpy()
    pred_masks = pred
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

# -----------------------------------------------------------------------------------------


def rle_decode(mask_rle, shape=(520, 704)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) 
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run():
    best_model=(
        #{'file': 'tf_moreEpoch_model_0005807.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.317,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_diou_model_0006049.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.317,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_scheduler_model_0007017.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.317,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_lr00025_model_0009437.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        # 'LB score': 0.318,
        # 'ths':[.25, .45, .65]
        #},
        #{'file': 'gcnone_tf_model_0011131.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.316,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tfsche_againtf_sche_gc15_0023957.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.319,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'cleaned_model_0000241.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.319,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'cleaned_model_0001451.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.319,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_cleaned_lr0006_model_0004113.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.318,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_origin0006_model_0006775.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        #'LB score': 0.318,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_live0006_origin00055_model_0000725.pth','config_name':'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml',
        # 'LB score': 0.321,
        # 'ths':[.25, .45, .65]
        #},
        #{'file': 'tf_msk101_model_0004113.pth','config_name':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        #'LB score': 0.3,
        #'ths':[.25, .45, .65]
        #},
        #{'file': 'tfx2_msk101_model_0001451.pth','config_name':'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        # 'LB score': 0.3,
        # 'ths':[.25, .45, .65]
        #},
    )


    #config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    mdl_path = "./final_models"
    DATA_PATH = "../data"
    MODELS = []
    BEST_MODELS =[]
    THSS = []
    ID_TEST = 0
    SUBM_PATH = f'{DATA_PATH}/test'
    SINGLE_MODE = False
    NMS = True
    MIN_PIXELS = [75, 150, 75]
    IOU_TH = .3
    for b_m in best_model:
        model_name=b_m["file"]
        model_ths=b_m["ths"]
        config_name=b_m["config_name"]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_name))
        cfg.INPUT.MASK_FORMAT = 'bitmask'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
        cfg.MODEL.WEIGHTS = f'{mdl_path}/{model_name}'
        cfg.TEST.DETECTIONS_PER_IMAGE = 4000#4000#1000
        #cfg.TEST.AUG.ENABLED = True
        #cfg.TEST.AUG.MIN_SIZES = (400, 500, 512, 600, 700, 800)
        #cfg.TEST.AUG.MAX_SIZE = 1333
        #cfg.TEST.AUG.FLIP = False#True
        cfg.SEED = 225
        MODELS.append(DefaultPredictor(cfg))
        BEST_MODELS.append(model_name)
        THSS.append(model_ths)
    print(f'all loaded:\nthresholds: {THSS}\nmodels: {BEST_MODELS}')

    
    register_coco_instances('sartorius_val',{},'../coco_anot/annotations_val.json', 
                            '../data/')
    val_ds = DatasetCatalog.get('sartorius_val')
    
    scorings = []
    subm_ids, subm_masks = [], []
    for item in tqdm.tqdm(val_ds):
        
        
        
        ## ensemble_preds
        img = cv2.imread(item['file_name'])
        classes = []
        scores = []
        bboxes = []
        masks = []
        for i, model in enumerate(MODELS):
            output = model(img)
            pred_classes = output['instances'].pred_classes.cpu().numpy().tolist()
            pred_class = max(set(pred_classes), key=pred_classes.count)
            take = output['instances'].scores >= THSS[i][pred_class]
            classes.extend(output['instances'].pred_classes[take].cpu().numpy().tolist())
            scores.extend(output['instances'].scores[take].cpu().numpy().tolist())
            bboxes.extend(output['instances'].pred_boxes[take].tensor.cpu().numpy().tolist())
            masks.extend(output['instances'].pred_masks[take].cpu().numpy())
        assert len(classes) == len(masks) , 'ensemble lenght mismatch'
        #return classes, scores, bboxes, masks
        
        
        ## nms_predictions
        shape=(520, 704)
        he, wd = shape[0], shape[1]
        boxes_list = [[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he]
                      for x in bboxes]
        scores_list = [x for x in scores]
        labels_list = [x for x in classes]
        nms_bboxes, nms_scores, nms_classes = nms(
            boxes=[boxes_list], 
            scores=[scores_list], 
            labels=[labels_list], 
            weights=None,
            iou_thr=IOU_TH
        )
        nms_masks = []
        for s in nms_scores:
            nms_masks.append(masks[scores.index(s)])
        classes, scores, masks = zip(*sorted(zip(nms_scores, nms_classes, nms_masks), reverse=True))
        #return nms_classes, nms_scores, nms_masks
        
        targ = item
        pred_masks = list(masks)
        
        enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
        enc_targs = list(map(lambda x:x['segmentation'], targ['annotations']))
        ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, ious)
            p = tp / (tp + fp + fn)
            prec.append(p)
        
        scorings.append(np.mean(prec))
        
        
        ## ensemble_pred_masks
        shape=(520, 704)
        encoded_masks = []
        #pred_class = max(set(classes), key=classes.count)
        used = np.zeros(shape, dtype=int) 
        for i, mask in enumerate(masks):
            mask = mask * (1 - used)
            if mask.sum() >= MIN_PIXELS[pred_class]:
                used += mask
                encoded_masks.append(rle_encode(mask))
        #return result
        
        for enc_mask in encoded_masks:
            subm_ids.append(item['file_name'])
            subm_masks.append(enc_mask)
    print(f"Validation dataset MaP : {np.mean(scorings)}")

#     pd.DataFrame({
#         'id': subm_ids, 
#         'predicted': subm_masks
#     }).to_csv('submission.csv', index=False)

    
if __name__ == '__main__':
    run()