import pickle

import os
import copy
import numpy as np
import torch
import SharedArray as SA
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils
#from ...models.visualizers import PolyScopeVisualizer
from ...config import cfg_from_yaml_file, cfg
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import (
    points_in_boxes_cpu
)

class PointConstrast(object):
    def __init__(self, root_path, sampler_cfg, logger=None):
        
