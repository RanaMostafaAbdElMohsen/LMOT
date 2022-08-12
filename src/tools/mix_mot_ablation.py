
import json
import os

mot_json = json.load(open('/home/rana/Downloads/MOT17/annotations/train_half.json','r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = 'mot17_train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']

print('mot17')
print('Image List: ', len(img_list))
print('Ann List: ', len(ann_list))
print('Video List: ', len(video_list))

max_img = 10000
max_ann = 140000
max_video = 22

mot15_json = json.load(open('/home/rana/Downloads/MOT15/annotations/train.json','r'))
img_id_count = 0
for img in mot15_json['images']:
    img_id_count += 1
    img['file_name'] = 'mot15_train/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in mot15_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({
    'id': max_video,
    'file_name': 'mot15_train'
})

print('mot15')
print('Image List: ', len(img_list))
print('Ann List: ', len(ann_list))
print('Video List: ', len(video_list))

max_img = 30000
max_ann = 200000
max_video = 23

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

print('crowdhuman_train')
print('Image List: ', len(img_list))
print('Ann List: ', len(ann_list))
print('Video List: ', len(video_list))

max_img = 80000
max_ann = 1000000
max_video = 24

ethz_json = json.load(open('/home/rana/Downloads/ETHZ/annotations/train.json','r'))
img_id_count = 0
for img in ethz_json['images']:
    img_id_count += 1
    img['file_name'] = 'ethz_train/' + img['file_name'][5:]
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in ethz_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({
    'id': max_video,
    'file_name': 'ethz'
})

print('ETHZ')
print('Image List: ', len(img_list))
print('Ann List: ', len(ann_list))
print('Video List: ', len(video_list))

max_img = 100000
max_ann = 1500000
max_video = 24

cp_json = json.load(open('/home/rana/Downloads/Cityscapes/annotations/train.json','r'))
img_id_count = 0
for img in cp_json['images']:
    img_id_count += 1
    img['file_name'] = 'cp_train/' + img['file_name'][11:]
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in cp_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('Cityscapes')

video_list.append({
    'id': max_video,
    'file_name': 'cityperson'
})

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('/home/rana/Downloads/mix_mot_ablation/annotations/train.json','w'))