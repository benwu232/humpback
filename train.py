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
from dataset import *


def run(config):
    name = f'{config.task.name}-{config.model.backbone}-{config.loss.name}'

    df, trn_idxes, val_idxes, labels, label2idx = prepare_df(config)
    trn_ds = WhaleDataSet(config, df, trn_idxes, labels, label2idx)
    next(iter(trn_ds))



    df = pd.read_csv(LABELS)
    change_new_whale(df, new_whale_id)
    df = filter_df(df, n_new_whale=config.train.new_whale, new_whale_id=new_whale_id)
    df_fname = df.set_index('Image')
    #val_idxes = split_data_set(df, seed=1)
    #val_idxes = split_whale_idx(df, new_whale_method=(config.train.new_whale!=0), seed=97)
    #val_idxes = split_whale_idx(df, new_whale_method=0, seed=97)
    val_idxes = split_whale_idx(df, new_whale_method=config.train.new_whale, seed=97)

    #scoreboard = load_dump(pdir.models)
    scoreboard_file = pdir.models/f'scoreboard-{name}.pkl'
    scoreboard = Scoreboard(scoreboard_file,
                            config.scoreboard.len,
                            sort=config.scoreboard.sort)

    batch_size = config.train.batch_size
    n_process = config.n_process
    vision_trans = get_transforms(do_flip=False,
                                  p_lighting=0.9, max_lighting=0.6,
                                  max_rotate=18,
                                  max_zoom=1.2,
                                  p_affine=0.9,
                                  xtra_tfms=[
                                      RandTransform(tfm=TfmCoord (jitter), kwargs={'magnitude': 0.01}),
                                      RandTransform(tfm=TfmCoord (symmetric_warp), kwargs={'magnitude': (-0.2, 0.2)}),
                                      RandTransform(tfm=TfmPixel (cutout), kwargs={'n_holes': (1, 4), 'length': (5, 20)}),

                                  ],
                                  )
    data = (
        ImageList
            .from_df(df, TRAIN, cols=['Image'])
            #.filter_by_func(lambda fname: df_fname.at[Path(fname).name, 'Count'] > 3)
            .split_by_idx(val_idxes)
            #.split_by_valid_func(lambda path: path2fn(path) in val_fns)
            #.label_from_func(lambda path: fn2label[path2fn(path)])
            .label_from_df(cols='Id')
            .add_test(ImageList.from_folder(TEST))
            .transform(vision_trans, size=SZ, resize_method=ResizeMethod.SQUISH)
            .databunch(bs=batch_size, num_workers=n_process, path=pdir.root)
            #.normalize(imagenet_stats)
    )

    sampler = None
    if config.train.balance:
        class_count = [0] * len(data.classes)
        for k, cls in enumerate(data.train_ds.y.items):
            class_count[cls] += 1
        class_count[-1] = 4  #set new_whale

        class_sample_count = np.array([0] * len(data.train_ds))
        for k, cls in enumerate(data.train_ds.y.items):
            class_sample_count[k] = class_count[cls]

        weights = 1 / class_sample_count
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size*config.train.batches_per_epoch)

    train_dl = DataLoader(
        data.train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        sampler=sampler,
        num_workers=config.n_process
    )

    val_dl = DataLoader(
        data.valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_process
    )

    test_dl = DataLoader(
        data.test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.n_process
    )

    data_bunch = ImageDataBunch(train_dl, val_dl, test_dl=test_dl, device=device)
    #data_bunch.add_tfm(normalize)
    data_bunch = data_bunch.normalize(imagenet_stats)

    backbone = get_backbone(config)
    loss_fn = get_loss_fn(config)

    method = config.train.method
    if method in [1, 2]:
        true_wd = True
    else:
        true_wd = False

    if config.model.head == 'MixHead':
        head = MixHead
    elif config.model.head == 'CosHead':
        head = CosHead

    learner = cnn_learner(data_bunch,
                          backbone,
                          loss_func=loss_fn,
                          custom_head=head(config),
                          init=None,
                          true_wd=true_wd,
                          path=pdir.root,
                          metrics=[acc_with_unknown, mapk_with_unknown]
                          #metrics=[accuracy, map5, mapkfast])
                          )
    if config.train.new_whale != 0:
        learner.data.classes[-1] = 'new_whale'
    #learner.to_fp16()
    learner.clip_grad(2.)

    cb_save_model = SaveModelCallback(learner, every="epoch", name=name)
    cb_early_stop = EarlyStoppingCallback(learner, min_delta=1e-4, patience=30)
    cb_cal_map5 = CalMap5Callback(learner)
    cb_scoreboard = ScoreboardCallback(learner,
                                       monitor='val_loss',
                                       scoreboard=scoreboard,
                                       config=config,
                                       mode=config.scoreboard.mode)
    #cbs = [cb_cal_map5, cb_scoreboard, cb_early_stop]
    #cbs = [cb_scoreboard, cb_early_stop]
    cbs = [cb_scoreboard]#, cb_cal_map5]

    model_file = ''
    if config.model.pretrain:
        if len(scoreboard) and scoreboard[0]['file'].is_file():
            model_file = scoreboard[0]['file'].name[:-4]
        elif (pdir.models/f'{name}-coarse.pth').is_file():
            model_file = f'{name}-coarse'

        #model_file = 'CosNet-densenet121-MixLoss-coarse'
        #model_file = 'densenet121-82'
        if model_file:
            print(f'loading {model_file}')
            learner.load(model_file, with_opt=True)
            #cur_epoch = int(re.search(r'-(\d+)$', model_file).group(1))

    if method == 1:
        #coarse stage
        if model_file == '':
            #learner.load(f'{name}-coarse')
            learner.fit_one_cycle(15, 1e-2)#, callbacks=cbs)
            fname = f'{name}-coarse'
            print(f'saving to {fname}')
            learner.save(fname)

            print('LR finding ...')
            learner.lr_find()
            learner.recorder.plot()
            plt.savefig('lr_find.png')

        # Fine tuning
        #learner.to_fp16()
        learner.clip_grad()
        learner.unfreeze()

        max_lr = 1e-3
        lrs = [max_lr/100, max_lr/10, max_lr]

        learner.fit_one_cycle(config.train.n_epoch, lrs, callbacks=cbs)

    elif method == 2:
        learner.clip_grad()
        learner.unfreeze()
        max_lr = 1e-1
        lrs = [max_lr/100, max_lr/10, max_lr]
        learner.fit_one_cycle(config.train.n_epoch, lrs, callbacks=cbs)

    elif method == 3:
        learner.fit_one_cycle(5, 1e-2)#, callbacks=cbs)
        learner.clip_grad()
        learner.unfreeze()
        max_lr = 1e-3
        lrs = [max_lr/100, max_lr/10, max_lr]
        learner.fit(config.train.n_epoch, lrs, callbacks=cbs)




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


