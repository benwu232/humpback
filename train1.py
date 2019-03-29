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


def run(config):
    name = f'{config.task.name}-fine'

    df0 = pd.read_csv(LABELS)
    df = filter_df(df0, n_new_whale=False, more_than=1)
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

    net = CosNet(config)
    learner = Learner(data,
                      CosNet(config),
                      #loss_func=nn.CrossEntropyLoss(),
                      loss_func=ArcFaceLoss(radius=config.model.radius),
                      path='../'
                      )

    #learner.split([learner.model.body[:6], learner.model[6:], learner.model.head, learner.model.metric])
    learner.clip_grad()

    #learner.load(f'{name}-coarse')
    learner.fit_one_cycle(20, 1e-2)
    learner.save(f'{name}-coarse')



    print('LR finding ...')
    learner.lr_find()
    learner.recorder.plot()
    plt.savefig('lr_find.png')

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


