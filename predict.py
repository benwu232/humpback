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
    name = f'{config.task.name}-{config.model.backbone}-{config.loss.name}'

    df = pd.read_csv(LABELS)
    #change_new_whale(df, 'z_new_whale')
    df = filter_df(df, n_new_whale=111)
    df_fname = df.set_index('Image')
    val_idxes = split_data_set(df, seed=1)

    #scoreboard = load_dump(pdir.models)
    scoreboard_file = pdir.models/f'scoreboard-{name}.pkl'
    sb_len = config.scoreboard.len
    scoreboard = Scoreboard(scoreboard_file, sb_len, sort='dec')

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

    if len(scoreboard) and scoreboard[0]['file'].is_file():
        model_file = scoreboard[0]['file'].name[:-4]
    else:
        model_file = config.train.pretrained_file
    #model_file = 'densenet121-27'
    print(f'loading {model_file} ...')
    learner.load(model_file)

    preds, _ = learner.get_preds(DatasetType.Test)#, n_batch=20)
    probs = F.softmax(preds, dim=1)
    probs, tops = probs.topk(5, dim=1)

    tops = tops.cpu().numpy()
    test_df = pd.read_csv(pdir.data/'sample_submission.csv')
    test_df = test_df.set_index('Image')
    with tqdm.tqdm(total=len(tops)) as pbar:
        for ri, class_idxes in enumerate(tops):
            fname = learner.data.test_ds.x.items[ri].name
            row = ''
            for class_idx in class_idxes:
                row += learner.data.classes[class_idx]
                row += ' '
                pass
            test_df.at[fname, 'Id'] = row
            pbar.update(100)
    sub_file = f'../submission/{name}.csv'
    print(f'write to {sub_file}')
    test_df.to_csv(sub_file)


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


