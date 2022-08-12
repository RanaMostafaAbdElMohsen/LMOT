from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.coco import COCO
from .datasets.kitti import KITTI
from .datasets.coco_hp import COCOHP
from .datasets.mot import MOT
from .datasets.nuscenes import nuScenes
from .datasets.crowdhuman import CrowdHuman
from .datasets.kitti_tracking import KITTITracking
from .datasets.custom_dataset import CustomDataset
from .datasets.mot_ch import MOTCH
from .datasets.mix_mots import MIXMOTS
from .datasets.mot_20_ch import MOT20CH
from .datasets.mix_mots_ablation import MIXMOTSABLATION
from .datasets.mot_ablation import MOTABLATION

dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'mot': MOT,
  'nuscenes': nuScenes,
  'crowdhuman': CrowdHuman,
  'kitti_tracking': KITTITracking,
  'mot_ch' : MOTCH,
  'mix_mots' : MIXMOTS,
  'mot20_ch': MOT20CH,
  'mix_mots_ablation': MIXMOTSABLATION,
  'mot_ablation': MOTABLATION
}


def get_dataset(dataset):
  return dataset_factory[dataset]
