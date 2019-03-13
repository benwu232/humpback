#!/usr/bin/env python
# coding: utf-8

# ## This is my attempt to use siamese with gan idea

# In[1]:


use_kaggle = 0
if use_kaggle:
    #!pip install git+https://github.com/fastai/fastai.git
    get_ipython().system('git clone https://github.com/benwu232/humpback')
    import sys
     # Add directory holding utility functions to path to allow importing utility funcitons
    #sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')
    sys.path.append('/kaggle/humback/')


# In[2]:


# Suppress annoying stderr output when importing keras.
import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')

sys.stderr = old_stderr

import random
from scipy.ndimage import affine_transform

import pickle
import numpy as np
from math import sqrt

# Determise the size of each image
from os.path import isfile
from PIL import Image as pil_image
from tqdm import tqdm_notebook

from pandas import read_csv
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy_thresh
from fastai.basic_data import *
from torch.utils.data import DataLoader, Dataset
from torch import nn
from fastai.callbacks.hooks import num_features_model, model_sizes
from fastai.layers import BCEWithLogitsFlat
from fastai.basic_train import Learner
from skimage.util import montage
import pandas as pd
from torch import optim
import re

if use_kaggle:
    from humpback.utils import *
else:
    from utils import *
    
from IPython.core.debugger import set_trace
#from functional import seq


# In[3]:


fastai.__version__


# In[4]:


root_path = Path('../input')
train_path = root_path/'train'
test_path = root_path/'test'
learn_path = Path('../')

USE_CUDA = torch.cuda.is_available()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if USE_CUDA else "cpu")


# In[5]:


#name = f'siamese_densenet121_224'
#arch = models.densenet121
#name = f'siamese_resnet50_224'
#arch = models.resnet50
name = f'siamese_resnet34_224'
arch = models.resnet34
#name = f'siamese_resnet18_224'
#arch = models.resnet18

im_size = 224
train_batch_size = 64
val_batch_size = 128
if use_kaggle:
    dl_workers = 0
else:
    dl_workers = 6
SEED=0
emb_len = 256


# In[6]:


df0 = pd.read_csv('../input/train.csv')
change_new_whale(df0, new_whale_id)
df_new = df0[df0.Id == new_whale_id]
df_known = df0[df0.Id != new_whale_id]
train_list, val_list = split_whale_set(df0, nth_fold=0, new_whale_method=1, seed=1, new_whale_id=new_whale_id)

im_count = df0[df0.Id != new_whale_id].Id.value_counts()
im_count.name = 'sighting_count'
ex_df = df0.join(im_count, on='Id')

path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
fn2label = {row[1].Image: row[1].Id for row in df0.iterrows()}
class_dict = make_whale_class_dict(df0)
file_lut = df0.set_index('Image').to_dict()


# In[8]:


im_tfms = get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2)

data = (
    ImageItemList
        # .from_df(df_known, 'data/train', cols=['Image'])
        .from_folder(train_path)
        # .split_by_idxs(train_item_list, val_item_list)
        .split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
        # .split_by_idx(val_list)
        # .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path2fn(str(path))])
        #.add_test(ImageItemList.from_folder(test_path))
        #.transform([None, None], size=im_size, resize_method=ResizeMethod.SQUISH)
        .transform(im_tfms, size=im_size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=train_batch_size, num_workers=dl_workers, path=root_path)
        #.normalize(imagenet_stats)
)

data.add_tfm(normalize_batch)

coach = Coach(learn=None, n_batch=313, batch_size=train_batch_size, ds_len=len(data.train_ds))

train_dl = DataLoader(
    SiameseGanDs(data.train_dl, coach.get_que()),
    #SiameseDs(data.train_dl),
    batch_size=train_batch_size,
    shuffle=False,
    #collate_fn=collate_siamese,
    num_workers=dl_workers
)

val_dl = DataLoader(
    SiameseDs(data.valid_dl),
    #SiameseDs(data.valid_dl),
    batch_size=val_batch_size,
    shuffle=False,
    #collate_fn=collate_siamese,
    num_workers=dl_workers
)


# In[9]:


#train_bunch = ImageDataBunch(train_dl, val_dl, collate_fn=collate_siamese, device=device)
train_bunch = ImageDataBunch(train_dl, val_dl, device=device)


# In[10]:


siamese = SiameseNet(emb_len=emb_len, arch=arch, forward_type='similarity', drop_rate=0.5)
#siamese = SiameseNetwork2(arch=arch)
siamese.to(device)


# In[11]:


learn = Learner(train_bunch,
                  siamese,
                  #enable_validate=True,
                  path=learn_path,
                  loss_func=BCEWithLogitsFlat(),
                  metrics=[avg_pos_sim, avg_neg_sim]
                  #metrics=[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)]
                  )

learn.coach_net = coach.coach_net
learn.coach_optim = torch.optim.Adam(learn.coach_net.parameters(), lr=1e-4)
learn.coach = coach
learn.coach.learn = learn
learn.enable_coach = True


learn.split([learn.model.cnn[:6], learn.model.cnn[6:], learn.model.fc])


# In[13]:


from fastai.callbacks import SaveModelCallback
cb_save_model = SaveModelCallback(learn, every="epoch", name=name)
cb_coach = CbCoachTrain(learn, n_train_batch=20)
cb_sims = CbSims(learn)
#cb_siamese_validate = SiameseValidateCallback(learn, txlog)
cbs = [cb_save_model, cb_coach, cb_sims]#, cb_siamese_validate]


# In[14]:


learn.freeze_to(-1)
learn.fit_one_cycle(3, callbacks=cbs)
learn.unfreeze()


# In[15]:


enable_lr_find = 0
if enable_lr_find:
    print('LR plotting ...')
    learn.lr_find()
    learn.recorder.plot()
    plt.savefig('lr_find.png')


# In[16]:


max_lr = 1e-4
lrs = [max_lr/20, max_lr/5, max_lr]
learn.fit_one_cycle(100, lrs, callbacks=cbs)


# In[ ]:





# In[ ]:





# In[ ]:




