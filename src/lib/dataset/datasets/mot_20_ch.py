from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..generic_dataset import GenericDataset
import os

class MOT20CH(GenericDataset):
  num_categories = 1
  num_classes =1 
  num_joints = 17
  default_resolution = [544, 960]
  max_objs = 256
  class_name = ['person']
  cat_ids = {1: 1}
  def __init__(self, opt, split):
    print('Using Mix of MOT20 and CrowdHuman')
    print("Using split: ", split)
    data_dir = os.path.join(opt.data_dir, 'mix_mot20_ch')
    img_dir = os.path.join(data_dir)
    ann_path = os.path.join(data_dir, 'annotations', 
      '{}.json').format(split)

    self.images = None
    # load image list and coco
    super().__init__(opt, split, ann_path, img_dir)

    self.num_samples = len(self.images)
    print('Loaded MOT20 and CrowdHuman dataset {} samples'.format(self.num_samples))
  
  def __len__(self):
    return self.num_samples

  def run_eval(self, results, save_dir):
    pass
