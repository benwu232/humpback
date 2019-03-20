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

name = 'softmax'

df0 = pd.read_csv(LABELS)
df = filter_df(df0, new_whale=False, more_than=1)
df_fname = df.set_index('Image')
val_idxes = split_whale_idx(df, nth_fold=0, new_whale_method=0, seed=1, new_whale_id='new_whale')

data = (
    ImageList
        .from_df(df, TRAIN, cols=['Image'])
        #.filter_by_func(lambda fname: df_fname.at[Path(fname).name, 'Count'] > 3)
        .split_by_idx(val_idxes)
        #.split_by_valid_func(lambda path: path2fn(path) in val_fns)
        #.label_from_func(lambda path: fn2label[path2fn(path)])
        .label_from_df(cols='Id')
        .add_test(ImageList.from_folder(TEST))
        .transform(get_transforms(do_flip=False), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)


learner = cnn_learner(data,
                      models.resnet18,
                      loss_func=nn.CrossEntropyLoss(),
                      lin_ftrs=[2048],
                      path='../'
                      )

learner.clip_grad()

#learner.load(f'{name}-coarse')
learner.fit_one_cycle(20, 1e-2)
learner.save(f'{name}-coarse')



print('LR finding ...')
learner.lr_find()
learner.recorder.plot()
plt.savefig('lr_find.png')


