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
colour1= (0, 255, 0)
colour3= (255, 0, 0)
colour2= (0 , 97, 230)
colour4= (0, 255, 255)
def draw_bbox(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    c=(255, 0, 255)
    # if int(bbox[4]) == 877:
    #       c=colour1
    # elif int(bbox[4]) == 873:
    #       c=colour2
    # elif int(bbox[4]) == 905:
    #       c=colour3
    # elif int(bbox[4]) == 907:
    #       c=colour4
    if int(bbox[4]) == 895:
          c=colour1
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
      c, 4, lineType=cv2.LINE_AA)
    ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    txt = '{}'.format(bbox[4])
    cv2.putText(img, txt, (int(bbox[0]), int(bbox[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                c, thickness=4, lineType=cv2.LINE_AA)

# if __name__ == '__main__':
#   seqs = os.listdir(GT_PATH)
#   # if SAVE_VIDEO:
#   #   save_path = sys.argv[1][:sys.argv[1].rfind('/res')] + '/video'
#   #   if not os.path.exists(save_path):
#   #     os.mkdir(save_path)
#   #   print('save_video_path', save_path)
#   for seq in sorted(seqs):
#     print('seq', seq)
#     # if len(sys.argv) > 2 and not sys.argv[2] in seq:
#     #   continue
#     if '.DS_Store' in seq:
#       continue
#     # if SAVE_VIDEO:
#     #   fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     #   video = cv2.VideoWriter(
#     #     '{}/{}.avi'.format(save_path, seq),fourcc, 10.0, (1024, 750))
#     seq_path = '{}/{}/'.format(GT_PATH, seq)
#     if IS_GT:
#       ann_path = seq_path + 'gt/gt.txt'
#     else:
#       ann_path = seq_path + 'det/det.txt'
#     anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
#     print('anns shape', anns.shape)
#     image_to_anns = defaultdict(list)
#     for i in range(anns.shape[0]):
#       if (not IS_GT) or (int(anns[i][6]) == 1 and float(anns[i][8]) >= 0.25):
#         frame_id = int(anns[i][0])
#         track_id = int(anns[i][1])
#         bbox = (anns[i][2:6] / RESIZE).tolist()
#         image_to_anns[frame_id].append(bbox + [track_id])
    
#     image_to_preds = {}
#     for K in range(1, len(sys.argv)):
#       image_to_preds[K] = defaultdict(list)
#       pred_path = sys.argv[K] + '/{}.txt'.format(seq)
#       try:
#         preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=',')
#       except:
#         preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=' ')
#       for i in range(preds.shape[0]):
#         frame_id = int(preds[i][0])
#         track_id = int(preds[i][1])
#         bbox = (preds[i][2:6] / RESIZE).tolist()
#         image_to_preds[K][frame_id].append(bbox + [track_id])
    
#     img_path = seq_path + 'img1/'
#     images = os.listdir(img_path)
#     num_images = len([image for image in images if 'jpg' in image])
    
#     for i in range(num_images):
#       frame_id = i + 1
      # file_name = '{}/img1/{:06d}.jpg'.format(seq, i + 1)
      # file_path = IMG_PATH + file_name
      # img = cv2.imread(file_path)
      # if RESIZE != 1:
      #   img = cv2.resize(img, (img.shape[1] // RESIZE, img.shape[0] // RESIZE))
      # for K in range(1, len(sys.argv)):
      #   img_pred = img.copy()
      #   draw_bbox(img_pred, image_to_preds[K][frame_id])
      #   cv2.imshow('pred{}'.format(K), img_pred)
      # draw_bbox(img, image_to_anns[frame_id])
      # cv2.imshow('gt', img)
      # cv2.waitKey()
      # if SAVE_VIDEO:
      #   video.write(img_pred)
    # if SAVE_VIDEO:
    #   video.release()

if __name__ == '__main__':

    image_to_preds = defaultdict(list)
    pred_path = "/home/rana/Downloads/MOT16/test/MOT16-08/gt/gt.txt"
    try:
      preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=',')
    except:
      preds = np.loadtxt(pred_path, dtype=np.float32, delimiter=' ')
    for i in range(preds.shape[0]):
      frame_id = int(preds[i][0])
      track_id = int(preds[i][1])
      bbox = (preds[i][2:6] / RESIZE).tolist()
      image_to_preds[frame_id].append(bbox + [track_id])
    
    img_path = "/home/rana/Downloads/MOT16/test/MOT16-08/" + 'img1/'
    images = os.listdir(img_path)
    num_images = len([image for image in images if 'jpg' in image])
    
    for i in range(num_images):
      frame_id = i + 1
      file_name = '{}{:06d}.jpg'.format(img_path, i + 1)
      file_path = file_name
      img = cv2.imread(file_path)
      # img = cv2.resize(img, (img.shape[1] // RESIZE, img.shape[0] // RESIZE))
      img_pred = img.copy()
      draw_bbox(img_pred, image_to_preds[frame_id])
      cv2.imshow('pred', img_pred)
      cv2.waitKey()
