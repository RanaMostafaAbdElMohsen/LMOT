import json
import os

mot_json = json.load(open('/home/rana/Downloads/MOT17/annotations/train.json','r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = 'mot_train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']

mot_samples = len(img_list)

print('mot17 with samples: ', mot_samples)

max_img = 16000
max_ann = 2000000
max_video = 10



mot_16_json = json.load(open('/home/rana/Downloads/MOT16/annotations/train.json','r'))
img_id_count = 0
for img in mot_16_json['images']:
    img_id_count += 1
    img['file_name'] = 'mot_16_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['prev_image_id'] + max_img
    img['next_image_id'] = img['next_image_id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = img['video_id'] + max_video
    img_list.append(img)
    
for ann in mot_16_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append(mot_16_json['videos'])

mot_16_samples = len(img_list) - (mot_samples)
print('MOT 16 with samples: ', mot_16_samples)

max_img = 22000
max_ann = 3000000
max_video = 10

crowdhuman_json = json.load(open('/home/rana/Downloads/crowdhuman/annotations/train.json','r'))
img_id_count = 0
for img in crowdhuman_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_train'
})

crowdhuman_samples = len(img_list) - mot_16_samples - mot_samples

print('crowdhuman_train with samples: ', crowdhuman_samples)

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list

json.dump(mix_json, open('/home/rana/Downloads/mix_mots/annotations/train.json','w'))
