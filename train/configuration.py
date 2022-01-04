import os
import random
import collections
import numpy as np
import pandas as pd

import torch
import torchvision

class CONFIG():
    BASE_PATH = '../data/'
    MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" #"Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    
    BATCH = 4#2
    LEARNING_LATE = 0.0005
    BATCH_PER_IMAGE = 128
    ITER_SIZE = 3000