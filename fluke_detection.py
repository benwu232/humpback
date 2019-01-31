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
#import cv2
import types

from utils import *

root_dir = Path('../input')
train_dir = root_dir/'train'
annot_dir = root_dir/'annotation'


j = json.load(open(f'{annot_dir}/annotations.json'))

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

fn2bbox = {jj['filename']: [[anno2bbox(jj)], ['fluke']] for jj in j}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
get_y_func = lambda o: fn2bbox[path2fn(o)]

#def get_y_func(o):
#    return fn2bbox[path2fn(o)]

idxs = np.arange(len(j))
np.random.seed(0)
np.random.shuffle(idxs)
val_idxs = idxs[:100]

val_j = [anno for i, anno in enumerate(j) if i in val_idxs]
trn_j = [anno for i, anno in enumerate(j) if i not in val_idxs]
len(trn_j), len(val_j)

pd.to_pickle([anno['filename'] for anno in val_j], f'{annot_dir}/val_fns_detection.pkl') # this will allow me to use the same validation set across NBs
val_fns = pd.read_pickle(f'{annot_dir}/val_fns_detection.pkl')

val_fns[0]

class StubbedObjectCategoryList(ObjectCategoryList):
    def analyze_pred(self, pred):
        return [pred.unsqueeze(0), torch.ones(1).long()]

data = (ObjectItemList.from_df(pd.DataFrame(data=list(fn2bbox.keys())), path=train_dir)
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(get_y_func, label_cls=StubbedObjectCategoryList)
        .transform(get_transforms(max_zoom=1, max_warp=0.05, max_rotate=0.05, max_lighting=0.2), tfm_y=True, size=(SZ,SZ), resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS)
        .normalize(imagenet_stats))

data.show_batch(rows=3, ds_type=DatasetType.Valid, figsize=(12,12))