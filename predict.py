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
from functools import partial
import torchvision
import pprint
from utils import *
from models import *
from dataset import *
from utils import *


def run(config):
    #global device
    #if 'device' in config:
    #    device = torch.device(config.device)

    new_whale_idx = 5004
    if config.train.use_flip:
        new_whale_idx = 10008

    acc_unknown = partial(cal_acc, new_whale_idx=new_whale_idx)
    mapk_unknown = partial(cal_mapk, new_whale_idx=new_whale_idx)
    #acc_known = partial(cal_acc, new_whale_idx=new_whale_idx, with_new_whale=False)
    #mapk_known = partial(cal_mapk, new_whale_idx=new_whale_idx, with_new_whale=False)

    name = f'{config.task.name}-{config.model.backbone}-{config.loss.name}'

    config.env.update(init_env(config))
    batch_size = config.train.batch_size
    trn_ds = WhaleDataSet(config, mode='train')
    val_ds = WhaleDataSet(config, mode='val')
    test_ds = WhaleDataSet(config, mode='test')

    trn_dl = DataLoader(
        trn_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        #sampler=sampler,
        pin_memory=True,
        num_workers=config.n_process
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.n_process
    )

    tst_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.n_process
    )

    #trn_dl, val_dl, tst_dl = map(lambda ts: DeviceDataLoader(*ts), zip([trn_dl, val_dl, tst_dl], [device] * 3) )
    data_bunch = ImageDataBunch(trn_dl, val_dl, test_dl=tst_dl, device=device)
    #data_bunch = DataBunch(trn_dl, val_dl, test_dl=tst_dl, device=device)
    #data_bunch.add_tfm(normalize)
    #data_bunch = data_bunch.normalize(imagenet_stats)

    #scoreboard = load_dump(pdir.models)
    scoreboard_file = config.env.pdir.models/f'scoreboard-{name}.pkl'
    scoreboard = Scoreboard(scoreboard_file,
                            config.scoreboard.len,
                            sort=config.scoreboard.sort)

    backbone = get_backbone(config)
    loss_fn = get_loss_fn(config, new_whale_idx=new_whale_idx)#, label_weight=data_bunch.train_ds.label_weight)

    method = config.train.method
    #if method in [1, 2]:
    #    true_wd = True
    #else:
    #    true_wd = False

    if config.model.head == 'MixHead':
        head = MixHead
    elif config.model.head == 'CosHead':
        head = CosHead

    learner = cnn_learner(data_bunch,
                          backbone,
                          loss_func=loss_fn,
                          custom_head=head(config),
                          init=None,
                          true_wd=config.train.true_wd,
                          wd=config.train.wd,
                          path=config.env.pdir.root,
                          metrics=[acc_unknown, mapk_unknown, acc_known, mapk_known]
                          #metrics=[accuracy, map5, mapkfast])
                          )
    if config.train.new_whale != 0:
        learner.data.classes[-1] = 'new_whale'
    #learner.to_fp16()
    learner.clip_grad(2.)
    loss_fn.model = learner.model

    cb_save_model = SaveModelCallback(learner, every="epoch", name=name)
    cb_early_stop = EarlyStoppingCallback(learner, min_delta=1e-4, patience=30)
    cb_cal_map5 = CalMap5Callback(learner)
    cb_scoreboard = ScoreboardCallback(learner,
                                       monitor='val_map',
                                       scoreboard=scoreboard,
                                       config=config,
                                       )
    #cbs = [cb_cal_map5, cb_scoreboard, cb_early_stop]
    #cbs = [cb_scoreboard, cb_early_stop]
    cbs = [cb_scoreboard]#, cb_cal_map5]

    model_file = ''
    if config.model.pretrain:
        if 'pretrained_file' in config.train:
            model_file = config.train.pretrained_file
        elif len(scoreboard) and scoreboard[0]['file'].is_file():
            model_file = scoreboard[0]['file'].name[:-4]
        elif (config.env.pdir.models/f'{name}-coarse.pth').is_file():
            model_file = f'{name}-coarse'

        #model_file = 'CosNet-known-densenet121-4'
        #model_file = 'CosNet-densenet121-MixLoss-coarse'
        #model_file = 'densenet121-82'
        if model_file:
            print(f'loading {model_file}')
            learner.load(model_file, with_opt=True)
            #cur_epoch = int(re.search(r'-(\d+)$', model_file).group(1))


    #learner.export(f'{config.task.name}-{config.model.backbone}.pkl')

    print('predicting...')
    preds, y = predict_mixhead(learner.model, learner.data.test_dl)

    new_whale = 5004
    threshold = 0.335
    threshold = -1

    top5_idxes = insert_new_whale(preds, threshold, new_whale)

    df_sub = pd.read_csv(config.env.pdir.data/'sample_submission.csv')
    df_sub = df_sub.set_index('Image')

    top5_idxes = top5_idxes.detach().cpu().numpy()

    for k, f in enumerate(learner.data.test_ds.test_list):
        #f = learner.data.test_ds.test_list[0]
        row = learner.data.test_ds.labels[top5_idxes[k, 0]]
        for col in range(1, 5):
            row += f' {learner.data.test_ds.labels[top5_idxes[k, col]]}'
        df_sub.at[f.name, 'Id'] = row

    sub_file = config.env.pdir.root/'submission'/f'{config.task.name}-{config.model.backbone}.csv'
    print(f'submission file: {sub_file}')
    df_sub.to_csv(sub_file)




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


