import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
from fastai.callbacks.hooks import num_features_model
from torch.nn import L1Loss
import pandas as pd
from torch import optim
import re
import json
import random
#import cv2
import types

from utils import *

root_dir = Path('./../input')
train_dir = root_dir/'train'
annot_dir = root_dir/'annotation'
old_train_dir = root_dir/'old/train'

def points2bbox(points):
    left, top = points[0]
    right, bottom = points[0]
    for x, y in points[1:]:
        left = min(left, x)
        top = min(top, y)
        right = max(right, x)
        bottom = max(bottom, y)
    return [top, left, bottom, right]


with open(annot_dir/'cropping.txt', 'rt') as f:
    data = f.read().split('\n')[:-1]
data = [line.split(',') for line in data]
old_annot = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data]

root_dir/'old/train'/old_annot[0][0]

points2bbox(old_annot[0][1])

def relative_path(original_path, relative_to=root_dir):
    return pathlib.PurePath(original_path).relative_to(relative_to)

old_fn2bbox = {relative_path(root_dir/'old/train'/anno[0]): [[points2bbox(anno[1])], ['fluke']] for anno in old_annot}



new_annot = json.load(open(f'{annot_dir}/annotations.json'))

SZ = 224
BS = 64
NUM_WORKERS = 6

def anno2bbox(anno):
    im_width, im_height = PIL.Image.open(f"../input/train/{anno['filename']}").size
    file = anno['filename']
    for anno in anno['annotations']:
        if anno['class'] == 'fluke':
            break
    #anno = anno['annotations'][0]
    #print(file, anno)
    return [
        np.clip(anno['y'], 0, im_height) / im_height * SZ,
        np.clip(anno['x'], 0, im_width) / im_width * SZ,
        np.clip(anno['y']+anno['height'], 0, im_height) / im_height * SZ,
        np.clip(anno['x']+anno['width'], 0, im_width) / im_width * SZ
    ]

fn2bbox = {relative_path(root_dir/'train'/jj['filename']): [[anno2bbox(jj)], ['fluke']] for jj in new_annot}
fn2bbox.update(old_fn2bbox)

get_y_func = lambda o: fn2bbox[relative_path(o)]

#def get_y_func(o):
#    return fn2bbox[relative_path(o)]


n_val = 300
fn_list = list(fn2bbox.keys())
random.seed(0)
random.shuffle(fn_list)
val_fns = fn_list[:n_val]
trn_fns = fn_list[n_val:]

'''
idxes = np.arange(len(fn2bbox))
np.random.seed(0)
np.random.shuffle(idxes)
val_idxes = idxes[:300]

val_j = [anno for i, anno in enumerate(j) if i in val_idxes]
trn_j = [anno for i, anno in enumerate(j) if i not in val_idxes]
len(trn_j), len(val_j)

#pd.to_pickle([anno['filename'] for anno in val_j], f'{annot_dir}/val_fns_detection.pkl') # this will allow me to use the same validation set across NBs
#val_fns = pd.read_pickle(f'{annot_dir}/val_fns_detection.pkl')
val_fns = [anno['filename'] for anno in val_j]
val_fns[0]
'''

class StubbedObjectCategoryList(ObjectCategoryList):
    def analyze_pred(self, pred):
        return [pred.unsqueeze(0), torch.ones(1).long()]

data = (ObjectItemList.from_df(pd.DataFrame(data=fn_list), path=root_dir)
        .split_by_valid_func(lambda path: relative_path(path) in val_fns)
        #.split_by_valid_func(val_func)
        .label_from_func(get_y_func, label_cls=StubbedObjectCategoryList)
        .transform(get_transforms(max_zoom=1, max_warp=0.05, max_rotate=0.05, max_lighting=0.2), tfm_y=True, size=(SZ,SZ), resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS)
        .normalize(imagenet_stats))

data.show_batch(rows=3, ds_type=DatasetType.Valid, figsize=(12,12))