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
import argparse
import torchvision
import pprint
from utils import *
from models import *


def run():

    df = pd.read_csv(LABELS)
    #change_new_whale(df, 'z_new_whale')
    df = filter_df(df, n_new_whale=111)
    df_fname = df.set_index('Image')
    val_idxes = split_data_set(df, seed=1)

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

    name = 'softmax'
    learn = create_cnn(data, models.resnet34, lin_ftrs=[2048], metrics=[accuracy, map5, mapkfast])
    learn.clip_grad();

    from fastai.callbacks import SaveModelCallback
    cb_save_model = SaveModelCallback(learn, every="epoch", name=name)
    cb_cal_map5 = CalMap5Callback(learn)
    #cb_siamese_validate = SiameseValidateCallback(learn, txlog)
    cbs = [cb_cal_map5]

    learn.fit_one_cycle(10, 1e-2, callbacks=cbs)
    learn.save('stage1')

    learn.unfreeze()

    max_lr = 1e-3
    lrs = [max_lr/100, max_lr/10, max_lr]

    learn.fit_one_cycle(24, lrs, callbacks=cbs)
    learn.save('stage2')

    #learner = cnn_learner(data, models.resnet18, metrics=[map5, mapkfast])



if __name__ == '__main__':
    run()


