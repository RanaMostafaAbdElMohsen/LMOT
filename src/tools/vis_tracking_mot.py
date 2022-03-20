import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path

GT_PATH = '/home/rana/Downloads/MOT16/test/'
IMG_PATH = GT_PATH
SAVE_VIDEO = True
RESIZE = 1
IS_GT = False
green= (0, 255, 0)
blue= (255, 0, 0)
orange= (0 , 97, 230)
yellow= (0, 255, 255)
purple= (255, 0, 255)
red= (0, 0, 255)
light_blue = (230, 216, 173)
colors =[red, green, blue, orange, yellow, purple, light_blue]

def draw_bbox(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    c = ids_colors[bbox[4]]
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
      c, 4, lineType=cv2.LINE_AA)
    ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    txt = '{}'.format(bbox[4])
    cv2.putText(img, txt, (int(bbox[0]), int(bbox[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                c, thickness=4, lineType=cv2.LINE_AA)

if __name__ == '__main__':

    image_to_preds = defaultdict(list)
    ids_colors ={}
    pred_path = "/home/rana/Downloads/MOT20/test/MOT20-06/gt/gt.txt"
    try:
      preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=',')
    except:
      preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=' ')
    
    color_index = 0
    for i in range(preds.shape[0]):
      frame_id = int(preds[i][0])
      track_id = int(preds[i][1])
      bbox = (preds[i][2:6] / RESIZE).tolist()
      image_to_preds[frame_id].append(bbox + [track_id])
      if track_id not in ids_colors:
        if color_index == len(colors):
          color_index=0
        ids_colors[track_id]= colors[color_index]
        color_index +=1
        
        
    img_path = "/home/rana/Downloads/MOT20/test/MOT20-06/" + 'img1/'
    images = os.listdir(img_path)
    num_images = len([image for image in images if 'jpg' in image])
    
    for i in range(num_images):
      frame_id = i + 1
      file_name = '{}{:06d}.jpg'.format(img_path, i + 1)
      file_path = file_name
      img = cv2.imread(file_path)
      # img = cv2.resize(img, (1700, img.shape[0] // RESIZE))
      img_pred = img.copy()
      draw_bbox(img_pred, image_to_preds[frame_id])
      cv2.imshow('pred', img_pred)
      cv2.waitKey()
