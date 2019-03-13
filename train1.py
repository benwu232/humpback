from fastprogress import master_bar, progress_bar
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import pretrainedmodels
from collections import OrderedDict
import math
from utils import *
import torchvision


df0 = pd.read_csv(LABELS)
df = prepro_df(df0, new_whale=False, more_than=1)

data = (
    ImageList
        .from_df(df, TRAIN, cols=['Image'])
        #.split_by_valid_func(lambda path: path2fn(path) in val_fns)
        #.label_from_func(lambda path: fn2label[path2fn(path)])
        #.add_test(ImageItemList.from_folder('data/test'))
        #.transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        #.databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        #.normalize(imagenet_stats)
)

data
pass
