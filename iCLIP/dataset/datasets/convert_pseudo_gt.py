import pandas as pd
import json
from PIL import Image
import time
import os
from tqdm import tqdm
import itertools
import argparse
import numpy as np
import json


res = pd.read_csv('/home/josmy/Code/iCLIP.data/output/resnet101_8x8f_denseserial/inference/ava_video_val_v2.2/result.csv', names=['id', 'timestamp', 'x1', 'x2', 'x3', 'x4', 'act', 'conf'])
ann_df = res[res.conf >= 0.5]
l = []
iter_num = len(ann_df)
for rows in tqdm(ann_df.itertuples(), total=iter_num, desc='Calculating  info'):
    _, movie_name, timestamp, x1, y1, x2, y2, action_id, conf = rows
    movie_infos = {}
    if movie_name not in movie_infos:
        movie_infos['movie_name'] = movie_name
        movie_infos['timestamp'] = timestamp
        img_path = os.path.join(movie_name, '{}.jpg'.format(timestamp))
        img_root = '/home/josmy/Code/iCLIP.data/AVA/keyframes/trainval'
        movie_infos['size'] = Image.open(os.path.join(img_root, img_path)).size
        width, height = movie_infos['size']
        
        box_w, box_h = x2 - x1, y2 - y1
        width = width
        height = height
        real_x1, real_y1 = x1 * width, y1 * height
        real_box_w, real_box_h = box_w * width, box_h * height
        area = real_box_w * real_box_h
        movie_infos['bbox'] = list(map(lambda x: round(x, 2), [real_x1, real_y1, real_box_w, real_box_h]))
        movie_infos['act'] = action_id
        movie_infos['conf'] = conf

    l.append(movie_infos)

df = pd.DataFrame(l)

df1 = pd.DataFrame()
df1['id_tms'] = df['movie_name'].astype(str) + '*'+ df['timestamp'].astype(str)
df1['bbox'] = df['bbox']
df1['act'] = df['act']
df1['conf'] = df['conf']



print(df1.head())

df = df1.groupby(['id_tms']).agg(lambda x: tuple(x)).applymap(list).reset_index()

d = dict(zip(df['id_tms'].values, zip(df['act'].values, df['bbox'].values, df['conf'].values)))


print(d['_7oWZq_s_Sk*903'])

with open('pseudo_gt.json', 'w') as fp:
    json.dump(d, fp)

# print(df.head())


# df.to_csv('pseudo_gt.csv', index=False, header=False)
