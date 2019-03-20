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
    name = f'{config.task.name}'

    df = pd.read_csv(LABELS)
    #change_new_whale(df, 'z_new_whale')
    df = filter_df(df, new_whale=111)
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

    #net = ArcNet(config)
    learner = Learner(data,
                      CosNet(config),
                      loss_func=nn.CrossEntropyLoss(),
                      #loss_func=ArcFaceLoss(radius=config.model.pars.radius, margin=config.model.pars.margin),
                      #loss_func=CosFaceLoss(radius=config.model.pars.radius, margin=0.0),
                      metrics=[accuracy, map5, mapkfast],
                      #metrics=[mapkfast],
                      path='../'
                      )

    learner.split([learner.model.body[:6], learner.model.body[6:], learner.model.head, learner.model.cos_sim])
    learner.freeze_to(-2)

    #coarse stage
    if not config.train.pretrained_file:
        learner.clip_grad(2.)
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

    # Fine tuning
    learner.clip_grad()
    learner.unfreeze()

    max_lr = 3e-3
    lrs = [max_lr/100, max_lr/10, max_lr, max_lr]
    from fastai.callbacks import SaveModelCallback
    cb_save_model = SaveModelCallback(learner, every="epoch", name=name)
    cb_cal_map5 = CalMap5Callback(learner)
    #cb_siamese_validate = SiameseValidateCallback(learn, txlog)
    cbs = [cb_save_model, cb_cal_map5]

    learner.fit_one_cycle(100, lrs, callbacks=cbs)




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


