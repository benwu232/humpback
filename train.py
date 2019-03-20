from fastprogress import master_bar, progress_bar
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks import *
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


def run(config):
    name = f'{config.task.name}'

    df = pd.read_csv(LABELS)
    #change_new_whale(df, 'z_new_whale')
    df = filter_df(df, new_whale=111)
    df_fname = df.set_index('Image')
    val_idxes = split_data_set(df, seed=1)

    batch_size = config.train.batch_size
    n_process = config.n_process
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
            .databunch(bs=batch_size, num_workers=n_process, path=pdir.root)
            .normalize(imagenet_stats)
    )

    backbone = get_backbone(config)
    loss_fn = get_loss_fn(config)
    learner = cnn_learner(data,
                          backbone,
                          loss_func=loss_fn,
                          custom_head=CosHead(config),
                          init=None,
                          #path=pdir.root,
                          metrics=[accuracy, map5, mapkfast])
    learner.clip_grad(2.)


    #coarse stage
    if not config.train.pretrained_file:
        #learner.load(f'{name}-coarse')
        learner.fit_one_cycle(5, 1e-2)
        fname = f'{name}-coarse'
        print(f'saving to {fname}')
        learner.save(fname)

        print('LR finding ...')
        learner.lr_find()
        learner.recorder.plot()
        plt.savefig('lr_find.png')
    else:
        learner.load(config.train.pretrained_file)
        #todo load the best model
        #scoreboard_file = pdir.models/f'scoreboard-{prefix}.pkl'
        #learner.load(f'{self.scoreboard[0][-1].name[:-4]}', purge=False)

    # Fine tuning
    learner.clip_grad()
    learner.unfreeze()

    max_lr = 1e-3
    lrs = [max_lr/100, max_lr/10, max_lr]
    cb_save_model = SaveModelCallback(learner, every="epoch", name=name)
    cb_early_stop = EarlyStoppingCallback(learner, min_delta=1e-4, patience=30)
    cb_cal_map5 = CalMap5Callback(learner)
    cb_model_pool = ModelPoolCallback(learner, config=config)
    #cbs = [cb_cal_map5, cb_model_pool, cb_early_stop]
    cbs = [cb_model_pool, cb_early_stop]

    learner.fit_one_cycle(config.train.n_epoch, lrs, callbacks=cbs)




def parse_args():
    description = 'Train humpback whale identification'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('Train humpback whale identification')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = load_config(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    #utils.prepare_train_directories(config)
    run(config)
    print('success!')


if __name__ == '__main__':
    main()


