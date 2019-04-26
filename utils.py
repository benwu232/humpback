import numpy as np
import torch
import pandas as pd
import torch
from copy import deepcopy
import datetime as dt
import time
import tqdm
import fastai
from fastai.vision import *
from fastai.basic_data import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks.hooks import num_features_model, model_sizes
import torchvision
import tensorboardX as tx
from common import *

#PATH = './'
#TRAIN = '../input/train/'
#TEST = '../input/test/'
#LABELS = '../input/train.csv'
#BOXES = '../input/bounding_boxes.csv'
#MODELS = './models'
SZ = 224
#SZ = 320
BS = 32
#NUM_WORKERS = 0

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
#device = torch.device("cpu")

new_whale_id = 'z_new_whale'

digit_len = 15

train_list_dbg = ['00050a15a.jpg', '0005c1ef8.jpg', '0006e997e.jpg', '833675975.jpg', '2fe2cc5c0.jpg', '2f31725c6.jpg',
                  '30eac8c9f.jpg', '3c4235ad2.jpg', '4e6290672.jpg', 'b2cabd9d8.jpg', '411ae328a.jpg', '204c7a64b.jpg',
                  '3161ae9b9.jpg', '399533efd.jpg', '108f230d8.jpg', '2c63ff756.jpg', '1eccb4eba.jpg', '20e7c6af4.jpg',
                  '23acb3957.jpg', '0d5777fc2.jpg', '2885ea466.jpg', '2d43e205a.jpg', '2c0892b4d.jpg', '847b16277.jpg',
                  '03be3723c.jpg', '1cc6523fc.jpg', '47d9ad983.jpg', '645300669.jpg', '77fef7f1a.jpg', '67947e46b.jpg',
                  '4bb9c9727.jpg', '166a9e05d.jpg', '2b95bba09.jpg', '6662f0d1c.jpg']
val_list_dbg = ['ffd61cded.jpg', 'ffdddcc0f.jpg', 'fffde072b.jpg', 'c5658abf0.jpg', 'cc6c1a235.jpg', 'ee87a2369.jpg',
                '8d601c3e1.jpg', 'b758f5366.jpg', 'b9aefbbc8.jpg', 'c5d1963ab.jpg', '910e6297a.jpg', 'bdb41dba8.jpg',
                'bf1eba5c2.jpg', 'cd650e905.jpg', 'b3ba738d7.jpg', 'c48bb3cbf.jpg', 'c5f7b85de.jpg', '79fac8c6d.jpg',
                'b94450f8e.jpg', 'bd3f7ba02.jpg', 'c938f9d6f.jpg', 'e92c45748.jpg', '9ae676cb0.jpg', 'a8d5237c0.jpg']



def find_new_whale_idx(classes):
    for k, cls in enumerate(classes):
        if cls == new_whale_id:
            return k


def change_new_whale(df, old_name='new_whale', new_name='z_new_whale'):
    for k in range(len(df)):
        if df.at[k, 'Id'] == old_name:
            df.at[k, 'Id'] = new_name


def filter_df(df0, n_new_whale=111, more_than=0, new_whale_id='new_whale'):
    '''
    :param df0:
    :param n_new_whale: 0: no new_whale; >0: keep number of new_whale; <0: keep all new_whale
    :param more_than: keep the whales which are more than more_than
    :return: processed dataframe
    '''
    df = deepcopy(df0)
    df_counted = df.groupby('Id').count()
    df_counted = df_counted.rename(columns={'Image': 'Count'})
    df = df.join(df_counted, on='Id')

    df_new_whale = df[df['Id'] == new_whale_id]
    df = df[df['Id'] != new_whale_id]
    df_new_whale = df_new_whale.sample(frac=1)
    if n_new_whale > 0:
        df_new_whale = np.split(df_new_whale, [n_new_whale], axis=0)[0]

    if more_than > 0:
        df = df[df['Count'] > more_than]

    if n_new_whale != 0:
        df = df.append(df_new_whale)

    df = df.reset_index()
    return df

def split_data_set1(df, seed=97):
    '''
    Split whale dataset to train and valid set
    seed: Random seed for shuffling
    '''
    n_known_whale = 400
    n_new_whale = 111
    np.random.seed(seed)
    val_idxes = []
    new_whale_idxes = []
    for name, group in df.groupby('Id'):
        group_num = len(group)
        group_idxes = group.index.tolist()
        if name == 'new_whale':
            np.random.shuffle(group_idxes)
            new_whale_idxes = list(group_idxes[:n_new_whale])
        elif group_num == 2 and len(val_idxes) < n_known_whale:
            val_idxes.append(random.choice(group_idxes))
        if len(val_idxes) > n_known_whale and new_whale_idxes:
            val_idxes.extend(new_whale_idxes)
            break
    return val_idxes

def split_data_set(df, seed=97):
    '''
    Split whale dataset to train and valid set
    seed: Random seed for shuffling
    '''
    n_known_whale = 1000
    n_new_whale = int(n_known_whale * 0.276)
    np.random.seed(seed)
    val_idxes = []
    new_whale_idxes = []
    for name, group in df.groupby('Id'):
        group_num = len(group)
        group_idxes = group.index.tolist()
        if group_num in [2, 3]:
            val_idxes.append(random.choice(group_idxes))
        elif name == new_whale_id:
            np.random.shuffle(group_idxes)
            new_whale_idxes = list(group_idxes[:n_new_whale])

    random.shuffle(val_idxes)
    val_idxes = val_idxes[:n_known_whale]
    val_idxes.extend(new_whale_idxes)
    random.shuffle(val_idxes)
    return val_idxes

def split_whale_idx(df, nth_fold=0, total_folds=5, new_whale_method=0, seed=1, new_whale_id='z_new_whale'):
    '''
    Split whale dataset to train and valid set based on k-fold idea.
    total_folds: number of total folds
    nth_fold: the nth fold
    new_whale_method: If 0, remove new_whale in all data sets; if 1, add new_whale to train/validation sets
    seed: Random seed for shuffling
    '''
    np.random.seed(seed)
    val_idxes = []
    for name, group in df.groupby('Id'):
        if new_whale_method == 0 and name == new_whale_id:
            continue
        group_num = len(group)
        group_idxes = group.index.tolist()
        if group_num > 1:
            np.random.shuffle(group_idxes)
            span = max(1, group_num // total_folds)
            val_i = group_idxes[nth_fold * span:(nth_fold + 1) * span]
            val_idxes.extend(val_i)

    return val_idxes


def split_whale_set(df, nth_fold=0, total_folds=5, new_whale_method=0, seed=1, new_whale_id='z_new_whale'):
    '''
    Split whale dataset to train and valid set based on k-fold idea.
    total_folds: number of total folds
    nth_fold: the nth fold
    new_whale_method: If 0, remove new_whale in all data sets; if 1, add new_whale to train/validation sets
    seed: Random seed for shuffling
    '''
    np.random.seed(seed)
    # list(df_known.groupby('Id'))
    train_list = []
    val_list = []
    val_idxes = []
    # df_known = df[df.Id!='new_whale']
    for name, group in df.groupby('Id'):
        # print(name, len(group), group.index, type(group))
        # if name == 'w_b82d0eb':
        #    print(name, df_known[df_known.Id==name])
        if new_whale_method == 0 and name == new_whale_id:
            continue
        group_num = len(group)
        images = group.Image.values
        if group_num > 1:
            np.random.shuffle(images)
            # images = list(images)
            span = max(1, group_num // total_folds)
            val_images = images[nth_fold * span:(nth_fold + 1) * span]
            train_images = list(set(images) - set(val_images))
            val_list.extend(val_images)
            train_list.extend(train_images)
        else:
            train_list.extend(images)

    return train_list, val_list


def gen_ref_ds(ds):
    "Generate reference dataset from original dataset without new_whale"
    ref_ds = deepcopy(ds)
    del_idxes = []
    for k, item in enumerate(ref_ds):
        if item[1].obj != 'new_whale':
            del_idxes.append(k)

    for k in reversed(del_idxes):
        del ref_ds.x[k]
        del ref_ds.y[k]

    return ref_ds


def plot_lr(self):
    if not in_ipynb():
        plt.switch_backend('agg')
    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.plot(self.iterations, self.lrs)
    if not in_ipynb():
        plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))


# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def acc_known(preds, targs, new_whale_idx=5004):
    #new_whale = torch.tensor(new_whale_idx).to(device)
    known_idxes = targs != new_whale_idx
    if len(known_idxes) == 0:
        return 0.0
    targs = targs[known_idxes].to(device)
    #print(len(preds[0][known_idxes]), len(known_idxes))
    softmax = preds[0][known_idxes].max(1)[1].view_as(targs)
    acc = (softmax == targs).sum().float() / len(targs)
    return acc

def cal_acc(preds, targs, new_whale_idx=5004, with_new_whale=True):
    new_whale = torch.tensor(new_whale_idx).to(device)
    targs = targs.to(device)
    softmax = preds[0].max(1)[1].view_as(targs)
    value = softmax
    if with_new_whale:
        #bin = (torch.sigmoid(preds[0]) > 0.5).long().sum(dim=-1)
        bin = (torch.sigmoid(preds[1]) >= 0.5).long()
        bin = bin.view_as(targs)
        value = torch.where(bin==0, new_whale, softmax)
    acc = (value == targs).sum().float() / len(targs)
    return acc

def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def map5(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    res = mapk([[t] for t in targs.cpu().numpy()], top_5.cpu().numpy(), 5)
    return torch.tensor(res)

def mapkfast(preds, targs, k=5):
    preds = preds[0][:, :5004]
    #print(preds.shape, targs.shape)
    top_5 = preds.topk(k, 1)[1]
    targs = targs.to(preds.device)
    scores = torch.zeros(len(preds), k).float().to(preds.device)
    for kk in range(k):
        scores[:,kk] = (top_5[:,kk] == targs).float() / float(kk+1)
    return scores.max(dim=1)[0].mean()

def mapk_known(preds, targs, k=5, new_whale_idx=5004):
    new_whale = torch.tensor(new_whale_idx).to(device)
    known_idxes = targs != new_whale_idx
    #print(preds.shape, targs.shape)
    top5 = preds[0][known_idxes].topk(k, 1)[1]
    targs = targs[known_idxes].to(preds[0].device)
    scores = torch.zeros(len(targs), k).float().to(preds[0].device)
    for kk in range(k):
        scores[:,kk] = (top5[:,kk] == targs).float() / float(kk+1)
    return scores.max(dim=1)[0].mean()

def cal_mapk(preds, targs, k=5, new_whale_idx=5004, with_new_whale=True):
    new_whale = torch.tensor(new_whale_idx).to(device)
    #print(preds.shape, targs.shape)
    top5 = preds[0].topk(k, 1)[1]
    if with_new_whale:
        #bin = (torch.sigmoid(bin_logits) > 0.5).long().sum(dim=-1)
        bin = (torch.sigmoid(preds[1]) >= 0.5).long()
        bin = bin.view(-1)
        for row in range(len(targs)):
            if bin[row] == 0:
                top5[row, 1:] = top5[row, :-1]
                top5[row, 0] = new_whale
    targs = targs.to(preds[0].device)
    scores = torch.zeros(len(preds[0]), k).float().to(preds[0].device)
    for kk in range(k):
        scores[:,kk] = (top5[:,kk] == targs).float() / float(kk+1)
    return scores.max(dim=1)[0].mean()

def insert_new_whale(preds, threshold, new_whale=5004):
    top5_values, top5_idxes = preds[0].topk(5, 1)
    for row in range(len(top5_idxes)):
        for col in range(5):
            if top5_values[row, col] < threshold:
                top5_idxes[row, col+1:] = top5_idxes[row, col:-1]
                top5_idxes[row, col] = new_whale
                break
    return top5_idxes

def cal_map5_thresh(preds, targs, threshold, new_whale=5004):
    targs = targs.to(device)
    top5_idxes = insert_new_whale(preds, threshold, new_whale)
    scores = torch.zeros(len(targs), 5).float().to(device)
    for kk in range(5):
        scores[:,kk] = (top5_idxes[:,kk] == targs).float() / float(kk+1)
    return scores.max(dim=1)[0].mean()

'''
def accuracy_with_unknown(preds, targs, k=5, unknown_idx=5004):
    #print(preds.shape, targs.shape)
    targs = targs.detach().cpu().numpy()
    preds_unknown = (torch.sigmoid(preds[:, 0]) > 0.5).view(-1).detach().cpu().numpy()
    preds_known = preds[:, 1:]
    preds_idxes = preds_known.max(dim=1)[1].detach().cpu().numpy()

    top_5 = preds_known.topk(k, 1)[1].detach().cpu().numpy()
    for ri in range(len(preds_known)):
        if preds_unknown[ri]:
            preds_idxes[ri] = unknown_idx

    return torch.tensor((preds_idxes == targs).mean())

def mapk_with_unknown(preds, targs, k=5, unknown_idx=5004):
    #print(preds.shape, targs.shape)
    targs = targs.detach().cpu().numpy()
    preds_unknown = (torch.sigmoid(preds[:, 0]) > 0.5).detach().cpu().numpy()

    preds_known = preds[:, 1:]
    top_5 = preds_known.topk(k, 1)[1].detach().cpu().numpy()
    for ri in range(len(preds_known)):
        if preds_unknown[ri]:
            top_5[ri, 1:] = top_5[ri, :-1]
            top_5[ri, 0] = unknown_idx

    scores = np.zeros([len(preds_known), k], dtype=np.float32)
    for kk in range(k):
        #scores[:,kk] = (top_5[:,kk] == targs).float() / float(kk+1)
        scores[:,kk] = (top_5[:,kk] == targs) / float(kk+1)
    return torch.tensor(scores.max(axis=1).mean())
'''

def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]


def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels


def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')



def make_whale_class_dict(df):
    whale_class_dict = {}
    for name, group in df.groupby('Id'):
        whale_class_dict[name] = group.Image.tolist()
    return whale_class_dict


class ImageItemListEx(ImageList):
    def __init__(self, *args, convert_mode='RGB', **kwargs):
        super().__init__(*args, convert_mode=convert_mode, **kwargs)

    def transform(self, tfms: Optional[Tuple[TfmList, TfmList]] = (None, None), **kwargs):
        "Set `tfms` to be applied to the xs of the train and validation set."
        if not tfms: return self
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self


class DataLoaderTrain(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()
        # self.mean, self.std = torch.tensor(imagenet_stats)
        self.stats = tensor(imagenet_stats).to(device)

    def proc_batch(self, b):
        data = {}
        for name in b[0]:
            if name == 'pos_valid_masks':
                data[name] = torch.tensor(b[0][name]).to(device)
                continue
            data[name] = []
            for im in b[0][name]:
                # im.show()
                transformed = im.apply_tfms(listify(self.tfms))
                # transformed.show()
                data[name].append(transformed.data)

            # data[name] = normalize(torch.stack(data[name]), self.mean, self.std).to(device)
            data[name] = normalize(torch.stack(data[name]).to(device), *self.stats)
        target = b[1][0].to(device)
        valid_mask = b[1][1].to(device)
        return data, (target, valid_mask)


class DataLoaderVal(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()
        self.stats = tensor(imagenet_stats).to(device)

    def proc_batch(self, b):
        if len(b) == 2:
            data = b[0].to(device)
            target = b[1].to(device)
        else:
            data = b.to(device)

        for k, item in enumerate(data):
            data[k] = torchvision.transforms.functional.normalize(item, self.stats[0], self.stats[1])

        if len(b) == 2:
            return data, target
        return data


def make_id_dict1(class_dict):
    id_dict = {}
    for k, id in enumerate(class_dict):
        if id not in id_dict:
            id_dict[id] = [k]
        else:
            id_dict[id].append(k)

    for id in id_dict:
        id_dict[id] = np.asarray(id_dict[id])

    return id_dict


def make_class_1_idx_dict(ds, ignore=[]):
    class_dict = OrderedDict()
    for k, y in enumerate(ds.y):
        if y.obj in ignore:
            continue
        if y.obj not in class_dict:
            class_dict[y.obj] = [k]

    for key in class_dict:
        class_dict[key] = np.asarray(class_dict[key])

    return class_dict


def make_class_idx_dict(ds, ignore=[]):
    class_idx_dict = {}
    class_len_dict = {}
    for k, y in enumerate(ds.y):
        if y.obj in ignore:
            continue
        if y.data not in class_idx_dict:
            class_idx_dict[y.data] = [k]
        else:
            class_idx_dict[y.data].append(k)

    for idx in class_idx_dict:
        class_idx_dict[idx] = np.asarray(class_idx_dict[idx])
        # class_idx_dict[idx] = torch.tensor(class_idx_dict[idx]).to(device)
        # class_len_dict[idx] = torch.tensor(len(class_idx_dict[idx])).to(device)

    return class_idx_dict


def make_class_idx_tensor(ds, ignore=[]):
    max_len = 100
    n_class = len(ds.y.classes)
    class_idx = [[] for _ in range(n_class)]
    class_idx_t = 0 - torch.ones([n_class, max_len], dtype=torch.int32)
    for k, y in enumerate(ds.y):
        if y.obj in ignore:
            continue
        class_idx[y.data].append(k)

    for k in range(n_class):
        if class_idx[k]:
            tmp = torch.tensor(class_idx[k])
            class_idx_t[k] = tmp.repeat(max_len)[:max_len]

    class_idx_t = class_idx_t.to(device)
    return class_idx_t


class SimpleDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, ds):
        'Initialization'
        self.ds = ds

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ds)

    def __getitem__(self, index):
        sample = self.ds[index]
        if isinstance(sample, tuple):
            # if len(sample) == 2:
            return sample[0], sample[1]
        else:
            return sample


class SiameseDs(torch.utils.data.Dataset):
    def __init__(self, dl):
        self.dl = dl
        self.ds = dl.dl.dataset
        self.idx2class = self.ds.y.items
        # self.id_dict = make_id_dict(self.whale_class_dict)
        # self.class_idx_dict, self.class_len_dict = make_class_idx_dict(ds)
        self.class_idx_dict = make_class_idx_dict(self.ds)
        self.class_idx = make_class_idx_tensor(self.ds)
        self.len = len(self.ds)
        self.new_whale = find_new_whale_idx(self.ds.y.classes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if np.random.choice([0, 1]):
            while True:
                idx_p, pos_valid = self.find_same(idx)
                if pos_valid:
                    im1, label1 = self.dl.__getitem__(idx)
                    im2, label2 = self.dl.__getitem__(idx_p)
                    assert label1.obj == label2.obj
                    return (im1, im2), 1
                else:
                    idx = np.random.randint(self.len)
        else:
            idx_n = self.find_different(idx)
            im1, label1 = self.dl.__getitem__(idx)
            im3, label3 = self.dl.__getitem__(idx_n)
            return (im1, im3), 0

    def find_same(self, idx):
        whale_class = self.idx2class[idx]
        class_members = self.class_idx_dict[whale_class]

        # for 1-class or new_whale, there is no same
        if len(class_members) == 1 or whale_class == self.new_whale:
            return idx, 0  # 0 means anchor-positive pair is not valid

        candidates = class_members[class_members != idx]  # same label but not same image
        candidate = np.random.choice(candidates)
        return candidate, 1

    def find_different(self, idx1, idx2=None):
        # already have idx2
        if idx2 is not None:
            return idx2

        whale_class = self.idx2class[idx1]
        while True:
            idx2 = np.random.randint(self.len)
            if idx2 != idx1 and self.idx2class[idx2] != whale_class:
                break
        return idx2


class SiameseDs3(SiameseDs):
    def __init__(self, dl):
        super().__init__(dl)

    def __getitem__(self, idx):
        while True:
            idx_p, pos_valid = self.find_same(idx)
            if pos_valid:
                im1, label1 = self.dl.__getitem__(idx)
                im2, label2 = self.dl.__getitem__(idx_p)
                assert label1.obj == label2.obj
                break
                # return [im1, im2], [label1, label2]
            else:
                idx = np.random.randint(self.len)

        idx_n = self.find_different(idx)
        im3, label3 = self.dl.__getitem__(idx_n)
        return [im1, im2, im3], [label1, label2, label3]


class SiameseDsTriplet(SiameseDs):
    def __init__(self, ds):
        self.ds = ds
        self.idx2class = ds.y.items
        # self.id_dict = make_id_dict(self.whale_class_dict)
        self.class_dict = make_class_idx_dict(ds)
        self.len = len(self.ds)
        self.new_whale = find_new_whale_idx(ds.y.classes)

    def __getitem__(self, idx):
        idx_a = idx
        idx_p, pos_valid = self.find_same(idx)
        idx_n = self.find_different(idx)
        return (self.ds[idx_a][0], self.ds[idx_p][0], self.ds[idx_n][0]), pos_valid


def collate_siamese(items):
    im1_list = []
    im2_list = []
    target_list = []

    # print('collate_siamese')
    for (im1, im2), target in items:
        im1_list.append(im1)
        im2_list.append(im2)
        target_list.append(target)

    batch = {}
    batch['im1'] = im1_list
    batch['im2'] = im2_list
    return batch, torch.tensor(target_list, dtype=torch.int32)


def collate_siamese_triplet(items):
    anchor_list = []
    pos_list = []
    neg_list = []
    pos_valid_masks = []

    for anchor, pos, neg, mask in items:
        anchor_list.append(anchor)
        pos_list.append(pos)
        neg_list.append(neg)
        pos_valid_masks.append(mask)

    batch = {}
    batch['anchors'] = anchor_list
    batch['pos_ims'] = pos_list
    batch['neg_ims'] = neg_list
    # batch['pos_valid_masks'] = np.stack(pos_valid_masks)
    batch_len = len(batch['anchors'])
    target = torch.cat((torch.ones(batch_len, dtype=torch.int32), torch.zeros(batch_len, dtype=torch.int32)))
    pos_mask = torch.tensor(pos_valid_masks, dtype=torch.int32)
    return batch, (target, pos_mask)


def cnn_activations_count(model, width, height):
    _, ch, h, w = model_sizes(create_body(model), (width, height))[-1]
    return ch * h * w


class SiameseGanDs(SiameseDs):
    def __init__(self, dl, coach_que):
        super().__init__(dl)
        self.batch_size = dl.batch_size
        self.coach_que = coach_que
        self.dl = dl
        self.img_list1 = []
        self.img_list2 = []
        self.targets = []
        self.pn_split = int(self.batch_size * 0.8)

    def __getitem__(self, idx):
        pn_idx = idx % self.batch_size
        # generate a batch of pos-neg pairs
        if pn_idx == 0:
            img_idxes1, img_idxes2 = self.coach_que.get()
            self.img_list1 = []
            self.img_list2 = []
            self.targets = []
            for k, (img_idx1, img_idx2) in enumerate(zip(img_idxes1, img_idxes2)):
                if k >= self.pn_split:
                    break
                img1, target1 = self.dl.__getitem__(img_idx1)
                img2, target2 = self.dl.__getitem__(img_idx2)
                if target1 == target2:
                    continue
                self.img_list1.append(img1.data)
                self.img_list2.append(img2.data)
                self.targets.append(target1 == target2)

        if pn_idx < len(self.img_list1):
            return (self.img_list1[pn_idx], self.img_list2[pn_idx]), 0
        else:
            self.img_list1 = []
            self.img_list2 = []
            self.targets = []
            while True:
                pp_idx = np.random.randint(self.len)
                idx_p, pos_valid = self.find_same(pp_idx)
                if pos_valid:
                    im1, label1 = self.dl.__getitem__(pp_idx)
                    im2, label2 = self.dl.__getitem__(idx_p)
                    # print(label1.obj, label2.obj)
                    assert label1.obj == label2.obj
                    return (im1, im2), 1


class CoachNet1(nn.Module):
    def __init__(self, hidden=120, drop_rate=0.5):
        super().__init__()
        self.hidden = hidden
        self.drop_rate = drop_rate
        self.n_cat = 5005
        self.n_idx = 100
        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.net1_trunk = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.Dropout(self.drop_rate),
            nn.PReLU()
        )

        self.net1_cat = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden, self.n_cat - 1),
        )

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.net1_idx = nn.Linear(self.hidden + self.n_cat - 1, self.n_idx)

        self.net2_trunk = nn.Sequential(
            nn.Linear(self.n_idx + self.n_cat - 1, self.hidden),
            nn.Dropout(self.drop_rate),
            nn.PReLU()
        )

        self.net2_cat = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden, self.n_cat),
        )

        self.dropout2 = nn.Dropout(self.drop_rate)
        self.net2_idx = nn.Linear(self.hidden + self.n_cat, self.n_idx)

    def forward(self, batch_size):
        # noise = torch.randn([self.batch_size, 1], device=device)
        # noise = torch.rand([self.batch_size, 1], device=device, requires_grad=True)
        noise = torch.rand([batch_size, 1], device=device)
        trunk1 = self.net1_trunk(noise)
        net1_cat = self.net1_cat(trunk1)
        mid = torch.cat([trunk1, net1_cat], dim=1)
        mid = self.dropout1(mid)
        net1_idx = self.net1_idx(mid)

        tmp = torch.cat([net1_cat, net1_idx], dim=1)
        trunk2 = self.net2_trunk(tmp)
        net2_cat = self.net2_cat(trunk2)
        mid = torch.cat([trunk2, net2_cat], dim=1)
        mid = self.dropout2(mid)
        net2_idx = self.net2_idx(mid)
        return F.softmax(net1_cat, dim=1), F.softmax(net1_idx, dim=1), F.softmax(net2_cat, dim=1), F.softmax(net2_idx,
                                                                                                             dim=1)


class CoachNet2(nn.Module):
    def __init__(self, hidden=300, drop_rate=0.5):
        super().__init__()
        self.hidden = hidden
        self.drop_rate = drop_rate
        self.n_cat = 5005
        self.n_idx = 100
        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.net1_trunk = nn.Sequential(
            nn.Linear(64, self.hidden),
            nn.Dropout(self.drop_rate),
            nn.PReLU()
        )

        self.net1_cat = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden, self.n_cat - 1),
        )

        self.dropout1 = nn.Dropout(self.drop_rate)
        self.net1_idx = nn.Linear(self.hidden + self.n_cat, self.n_idx)

        self.net2_trunk = nn.Sequential(
            nn.Linear(self.n_idx + self.n_cat - 1, self.hidden),
            nn.Dropout(self.drop_rate),
            nn.PReLU()
        )

        self.net2_cat = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden, self.n_cat),
        )

        self.dropout2 = nn.Dropout(self.drop_rate)
        self.net2_idx = nn.Linear(self.hidden + self.n_cat, self.n_idx)

    def forward(self, batch_size):
        noise = torch.rand(64, device=device)
        # noise = torch.randn(1, device=device)
        trunk1 = self.net1_trunk(noise)
        net1_cat = self.net1_cat(trunk1)
        net1_cat = F.softmax(net1_cat, dim=0)
        net1_cat_topk, net1_cat_topk_idxes = net1_cat.topk(batch_size)
        cat1_sparse = onehot_enc(net1_cat_topk_idxes, self.n_cat - 1) * net1_cat

        mid = torch.cat([trunk1, net1_cat])
        mid = mid.expand(batch_size, mid.shape[0])
        mid = self.dropout1(mid)
        mid_ex = torch.cat([net1_cat_topk.view(-1, 1), mid], dim=1)
        net1_idx = self.net1_idx(mid_ex)
        net1_idx = F.softmax(net1_idx, dim=1)
        net1_idx_max_value, net1_idx_max_idx = net1_idx.max(dim=1)
        net1_idx_sparse = onehot_enc(net1_idx_max_idx, 100) * net1_idx_max_value.view(-1, 1)

        tmp = torch.cat([cat1_sparse, net1_idx_sparse], dim=1)
        trunk2 = self.net2_trunk(tmp)
        net2_cat = self.net2_cat(trunk2)
        net2_cat = F.softmax(net2_cat, dim=0)
        mid = torch.cat([trunk2, net2_cat], dim=1)
        mid = self.dropout2(mid)
        net2_idx = self.net2_idx(mid)
        net2_idx = F.softmax(net2_idx, dim=0)

        # print(net1_cat.shape, net1_idx.shape, net2_cat.shape, net2_idx.shape)

        return net1_cat_topk, net1_cat_topk_idxes, net1_idx, net2_cat, net2_idx


class CoachNet3(nn.Module):
    def __init__(self, ds_len, hidden=300, drop_rate=0.5):
        super().__init__()
        self.ds_len = ds_len
        self.hidden = hidden
        self.drop_rate = drop_rate
        self.n_cat = 5005
        self.n_idx = 100
        self.noise_dim = 100
        self.hidden1 = 300
        self.hidden2 = 100
        self.hidden3 = 100
        self.out_len = digit_len * 2
        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.pn_map = nn.Sequential(
            nn.Linear(digit_len, self.hidden1),
            nn.Dropout(self.drop_rate),
            nn.PReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.Dropout(self.drop_rate),
            nn.PReLU(),
            nn.Linear(self.hidden2, digit_len),
        )

    def forward(self, batch_size):
        noise = torch.rand((batch_size, self.noise_dim), device=device)
        logits = self.pn_map(noise)
        logits = torch.tanh(logits)
        digits = torch.sign(logits)
        digits = (digits + 1) / 2
        return digits


def onehot_enc(labels, n_class=10):
    onehot = torch.zeros([len(labels), n_class], dtype=torch.float32).to(device)
    # onehot.scatter_(dim=1, index=ids, src=1)
    onehot.scatter_(1, labels.view(-1, 1), 1.0)
    return onehot


class CoachNet(nn.Module):
    def __init__(self, ds_len, hidden=300, drop_rate=0.5):
        super().__init__()
        self.ds_len = ds_len
        self.hidden = hidden
        self.drop_rate = drop_rate
        self.n_cat = 5005
        self.n_idx = 100
        self.noise_dim = 100
        self.hidden1 = 300
        self.hidden2 = 100
        self.hidden3 = 100
        self.out_len = digit_len * 2
        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.pn_map = nn.Sequential(
            nn.Linear(digit_len, self.hidden1),
            nn.Dropout(self.drop_rate),
            nn.PReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.Dropout(self.drop_rate),
            nn.PReLU(),
            nn.Linear(self.hidden2, self.ds_len)
        )

    def forward(self, batch):
        logits = self.pn_map(batch)
        return F.softmax(logits, dim=1)


lock = torch.multiprocessing.Lock()


class Coach(object):
    def __init__(self, learn=None, batch_size=64, n_batch=10, ds_len=0):
        self.learn = learn
        self.ds_len = ds_len
        self.coach_net = CoachNet(ds_len)
        self.coach_net.to(device)
        self.batch_size = batch_size
        self.n_batch = n_batch
        self.que = torch.multiprocessing.Queue(maxsize=n_batch)

    def get_que(self):
        return self.que

    def gen_batchs1(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch

        self.coach_net.eval()
        with torch.no_grad():
            for _ in range(n_batch):
                batch = self.coach_net(self.batch_size)
                self.que.put(batch)

    def gen_batchs(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch

        self.coach_net.eval()
        with torch.no_grad():
            for k in range(n_batch):
                # if k % 100 == 0:
                #    print(k, n_batch)

                idxes1_bin, idxes1 = rand_batch(self.batch_size, self.ds_len, digit_len)
                softmax = self.coach_net(idxes1_bin)
                max_values2, idxes2 = softmax.max(dim=1)

                img_list1 = idxes1.tolist()
                img_list2 = idxes2.tolist()
                self.que.put([img_list1, img_list2], block=False)
        print(img_list1[:10])
        print(img_list2[:10])

    def gen_batchs1(self, n_batch=None):
        if n_batch is None:
            n_batch = self.n_batch

        self.coach_net.eval()
        with torch.no_grad():
            for k in range(n_batch):
                # if k % 100 == 0:
                #    print(k, n_batch)
                cat1_topk, cat1_topk_idx, cat1_idx, cat2, cat2_idx = self.coach_net(self.learn.data.train_dl.batch_size)
                # print((cat10.grad_fn, idx_cat10.grad_fn, cat20.grad_fn, idx_cat20.grad_fn))
                cat1_idx_argmax = cat1_idx.max(dim=1)[1]
                cat2_argmax = cat2.max(dim=1)[1]
                cat2_idx_argmax = cat2_idx.max(dim=1)[1]
                img_idx1 = self.learn.data.train_ds.class_idx[cat1_topk_idx, cat1_idx_argmax]
                img_idx2 = self.learn.data.train_ds.class_idx[cat2_argmax, cat2_idx_argmax]

                # if cat2 is new_whale, re-pick img_idx2
                for i, cat in enumerate(cat2_argmax):
                    if cat == self.learn.data.train_ds.new_whale:
                        img_idx2[i] = torch.tensor(
                            random.choice(self.learn.data.train_ds.class_idx_dict[self.learn.data.train_ds.new_whale]))

                img_list1 = img_idx1.tolist()
                img_list2 = img_idx2.tolist()
                self.que.put([img_list1, img_list2], block=False)
        print(img_list1[:10])
        print(img_list2[:10])


def linear_schedule(step, pars):
    "Linearly output value, end_step must greater than start_step"
    start_value = pars[0]
    end_value = pars[1]
    start_step = pars[2]
    end_step = pars[3]
    assert start_step <= end_step

    if step < start_step:
        return start_value
    elif step >= end_step:
        return end_value
    return start_value - (step - start_step) * (start_value - end_value) / (end_step - start_step)


#linear_decay = partial(linear_schedule, pars=(1.0, 0.05, 2, 12))


def rand_batch(batch_size, ds_len, digit_len):
    idxes = np.random.randint(0, ds_len, batch_size)
    idxes_bin = np.zeros((batch_size, digit_len), dtype=np.float32)
    for k in range(batch_size):
        bin_str = np.binary_repr(idxes[k], width=digit_len)
        for i in range(digit_len):
            idxes_bin[k, i] = int(bin_str[i])
    return torch.tensor(idxes_bin).to(device), idxes


class CbCoachTrain(fastai.callbacks.tracker.TrackerCallback):
    def __init__(self, learn, n_train_batch=5, schedule_pars=(1.0, 0.05, 0, 10)):
        super().__init__(learn)
        self.learn = learn
        self.schedule = partial(linear_schedule, pars=schedule_pars)
        self.coach_net = learn.coach_net
        self.coach_optim = learn.coach_optim
        self.coach = learn.coach
        base = [2 ** k for k in range(digit_len)]
        self.base = torch.tensor(base, dtype=torch.float32).to(device)
        self.ds_len = len(self.learn.data.train_ds)
        self.batch_size = self.learn.data.train_dl.batch_size
        self.n_train_batch = n_train_batch
        # self.batch_size = self.learn.data.train_dl.batch_size

    # def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    # def on_epoch_end(self, epoch, **kwargs: Any) -> None:
    def on_epoch_begin(self, epoch, **kwargs: Any) -> None:
        epsilon = self.schedule(epoch)
        # self.learn.data.train_ds.on_epoch_end()

        # lock discriminator
        # unlock coach net
        # train coach net
        # lock coach net
        # unlock discriminator
        if epoch >= 0 and self.learn.enable_coach:
            print(f'***************** epoch {epoch}, training coach_net **********************')
            self.coach_net.train()
            for k in range(self.n_train_batch):
                # noise = torch.randint((batch_size, self.noise_dim), device=device)
                idxes1_bin, idxes1 = rand_batch(self.batch_size, self.ds_len, digit_len)
                softmax = self.coach_net(idxes1_bin)
                max_values2, idxes2 = softmax.max(dim=1)

                img_list1 = []
                img_list2 = []
                targets = []
                for i in range(len(idxes1)):
                    img1, target1 = self.learn.data.train_dl.dl.dataset.dl.__getitem__(idxes1[i])
                    img2, target2 = self.learn.data.train_dl.dl.dataset.dl.__getitem__(idxes2[i])
                    img_list1.append(img1.data)
                    img_list2.append(img2.data)
                    targets.append(target1 == target2)
                targets = torch.tensor(targets, dtype=torch.int32)
                batch = [torch.stack(img_list1).to(device), torch.stack(img_list2).to(device)], targets
                batch = normalize_batch(batch)

                self.learn.model.eval()
                with torch.no_grad():
                    logits = self.learn.model(*batch[0])
                    sims = torch.sigmoid(logits)

                # criterion
                # penalize wrong targets
                sims[targets == 1] = -10.0
                self.coach_optim.zero_grad()
                loss = -sims.view_as(max_values2) * max_values2
                # loss = -sims * (1-s) * (1-cat1_idx.max(dim=1)[0]) * (1-cat2.max(dim=1)[0]) * (1-cat2_idx.max(dim=1)[0])
                loss = loss.mean()
                loss.backward()
                self.coach_optim.step()
                # print(img_idx1, img_idx2)
            pn_similarity = sims[targets == 0].mean()
            print(f'batch: {k}, coach_net loss: {loss.item()}, pn_similarity: {pn_similarity}')

        # clear the queue
        while not self.coach.que.empty():
            self.coach.que.get()
        # fill the queue
        self.coach.gen_batchs()
        print('********************************************************************************')

        return 0

    def on_epoch_begin1(self, epoch, **kwargs: Any) -> None:
        epsilon = self.schedule(epoch)
        # self.learn.data.train_ds.on_epoch_end()

        # lock discriminator
        # unlock coach net
        # train coach net
        # lock coach net
        # unlock discriminator
        if epoch >= 0:
            print(f'***************** epoch {epoch}, training coach_net **********************')
            self.coach_net.train()
            for k in range(10):
                cat1_topk, cat1_topk_idx, cat1_idx, cat2, cat2_idx = self.coach_net(self.learn.data.train_dl.batch_size)
                # print((cat10.grad_fn, idx_cat10.grad_fn, cat20.grad_fn, idx_cat20.grad_fn))
                cat1_idx_argmax = cat1_idx.max(dim=1)[1]
                cat2_argmax = cat2.max(dim=1)[1]
                cat2_idx_argmax = cat2_idx.max(dim=1)[1]
                # cat1 = cat1.detach()
                # idx_cat1 = idx_cat1.detach()
                # cat2 = cat2.detach()
                # idx_cat2 = idx_cat2.detach()
                img_idx1 = self.learn.data.train_ds.class_idx[cat1_topk_idx, cat1_idx_argmax]
                img_idx2 = self.learn.data.train_ds.class_idx[cat2_argmax, cat2_idx_argmax]

                # if cat2 is new_whale, re-pick img_idx2
                for i, cat in enumerate(cat2_argmax):
                    if cat == self.learn.data.train_ds.new_whale:
                        img_idx2[i] = torch.tensor(
                            random.choice(self.learn.data.train_ds.class_idx_dict[self.learn.data.train_ds.new_whale]))

                img_list1 = []
                img_list2 = []
                targets = []
                for i in range(len(img_idx1)):
                    img1, target1 = self.learn.data.train_dl.dl.dataset.dl.__getitem__(img_idx1[i])
                    img2, target2 = self.learn.data.train_dl.dl.dataset.dl.__getitem__(img_idx2[i])
                    img_list1.append(img1.data)
                    img_list2.append(img2.data)
                    targets.append(target1 == target2)
                targets = torch.tensor(targets, dtype=torch.int32)
                batch = [torch.stack(img_list1).to(device), torch.stack(img_list2).to(device)], targets
                batch = normalize_batch(batch)

                self.learn.model.eval()
                with torch.no_grad():
                    dists = self.learn.model(*batch[0])

                # criterion
                # penalize wrong targets
                dists[targets == 1] = 100.0
                self.coach_optim.zero_grad()
                loss = dists * (1 - cat1_topk) * (1 - cat1_idx.max(dim=1)[0]) * (1 - cat2.max(dim=1)[0]) * (
                            1 - cat2_idx.max(dim=1)[0])
                loss = loss.mean()
                loss.backward()
                self.coach_optim.step()
                # print(img_idx1, img_idx2)
            pn_loss = torch.relu(contrastive_neg_margin - dists)
            pn_loss = pn_loss[pn_loss != 0].mean()
            print(f'batch: {k}, coach_net loss: {loss.item()}, pn_loss: {pn_loss}')

        # clear the queue
        while not self.coach.que.empty():
            self.coach.que.get()
        # fill the queue
        self.coach.gen_batchs()
        print('********************************************************************************\n')

        return 0


class SiameseNet1(nn.Module):
    def __init__(self, emb_len=256, arch=models.resnet18, forward_type='similarity', diff_method=1, drop_rate=0.5):
        super().__init__()
        self.cnn = create_body(arch)
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Dropout(drop_rate),
                                  nn.PReLU(),
                                  nn.Linear(cnn_activations_count(arch, 1, 1) * 2, emb_len)
                                  )

        # self.fc1 = nn.Linear(cnn_activations_count(arch, width, height), emb_len)
        self.fc = nn.Linear(emb_len, 1)
        self.dropout = nn.Dropout(drop_rate)
        self.diff_method = diff_method
        if self.diff_method == 1:
            self.diff_vec = self.diff_vec1
        else:
            self.diff_vec = self.diff_vec2

        self.forward_type = forward_type
        if self.forward_type in ['similarity', 'sim']:
            self.forward = self.forward_sim
        elif self.forward_type in ['distance', 'dist']:
            self.forward = self.forward_dist

    def im2emb(self, im):
        x = self.cnn(im)
        x = self.head(x)
        return x

    def diff_vec1(self, emb1, emb2):
        return torch.abs(emb1 - emb2)

    def diff_vec2(self, emb1, emb2):
        return ((emb1 - emb2) ** 2) / (emb1 + emb2)

    def distance(self, emb1, emb2):
        return F.pairwise_distance(emb1, emb2)
        # return self.diff_vec1(emb1, emb2).mean()

    def similarity1(self, diff):
        logits = self.fc(diff)
        return torch.sigmoid(logits)

    def similarity(self, im1, im2):
        emb1 = self.im2emb(im1)
        emb2 = self.im2emb(im2)
        diff = self.diff_vec1(emb1, emb2)
        logits = self.fc(diff)
        return torch.sigmoid(logits)

    def forward_sim(self, im1, im2):
        emb1 = self.im2emb(im1)
        emb2 = self.im2emb(im2)
        diff = self.diff_vec1(emb1, emb2)
        # similarity = self.similarity(diff)
        x = self.dropout(diff)
        logits = self.fc(x)
        return logits

    def forward_dist(self, im1, im2):
        x1 = self.im2emb(im1)
        x2 = self.im2emb(im2)
        dist = self.distance(x1, x2)
        return dist


class SiameseNet(nn.Module):
    def __init__(self, emb_len=256, arch=models.densenet121, n_slice=16, drop_rate=0.5):
        super().__init__()
        self.cnn = create_body(arch, cut=-1)
        # model = arch(pretrained=True)
        # self.cnn = nn.DataParallel(model.features[:-1])
        self.n_slice = n_slice

        # self.head = nn.Sequential(AdaptiveConcatPool2d(),
        #                          Flatten(),
        #                          nn.Dropout(drop_rate),
        #                          nn.PReLU(),
        #                          nn.Linear(cnn_activations_count(arch, 1, 1)*2, emb_len)
        #                          )
        n_filters = cnn_activations_count(arch, 1, 1) * 2
        self.head = nn.DataParallel(nn.Sequential(AdaptiveConcatPool2d(), Flatten(),
                                                  nn.BatchNorm1d(n_filters), nn.Dropout(drop_rate),
                                                  nn.Linear(n_filters, 1024),
                                                  nn.PReLU(),
                                                  nn.BatchNorm1d(1024), nn.Dropout(drop_rate),
                                                  nn.Linear(1024, emb_len)))
        self.metric = nn.DataParallel(Metric(emb_len))

    def forward(self, x):
        x = self.head(self.cnn(x))
        sz = x.shape[0]
        x1 = x.unsqueeze(1).expand((sz, sz, -1))
        x2 = x1.transpose(0, 1)
        # matrix of all vs all differencies
        d = (x1 - x2).view(sz * sz, -1)
        return self.metric(d)

    def cal_emb(self, x):
        return self.head(self.cnn(x))

    def cal_dist(self, x0, x):
        d = (x - x0)
        return self.metric(d)


class SiameseNetwork2(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__()
        self.cnn = create_body(arch)
        self.emb_len = 256
        drop_rate = 0.5
        self.dropout = nn.Dropout(drop_rate)
        self.emb = nn.Linear(num_features_model(self.cnn) * 2, self.emb_len)
        self.fc = nn.Linear(self.emb_len, 1)

    def im2emb(self, im):
        x = self.cnn(im)
        x = AdaptiveConcatPool2d()(x)
        x = Flatten()(x)
        x = self.dropout(x)
        x = self.emb(x)
        return x

    def forward(self, im1, im2):
        x1 = self.im2emb(im1)
        x2 = self.im2emb(im2)
        dl = self.diff(x1, x2)
        dropout = self.dropout(dl)
        out = self.fc(dropout)
        return out

    def diff(self, x1, x2):
        return (x1 - x2).abs()

    def similarity(self, x1, x2):
        dl = self.diff(x1, x2)
        logit = self.fc(dl)
        return torch.sigmoid(logit)

    def distance(self, emb1, emb2):
        return F.pairwise_distance(emb1, emb2)


# modified Radek's net not using seq
# similarity
class SiameseNetwork(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__()
        self.cnn = create_body(arch)
        self.fc = nn.Linear(num_features_model(self.cnn), 1)

    def im2emb(self, im):
        x = self.cnn(im)
        x = self.process_features(x)
        return x

    def forward(self, im1, im2):
        x1 = self.im2emb(im1)
        x2 = self.im2emb(im2)
        dl = self.distance(x1, x2)
        out = self.fc(dl)
        return out

    def process_features(self, x):
        return x.reshape(*x.shape[:2], -1).max(-1)[0]

    def distance(self, x1, x2):
        return (x1 - x2).abs()

    def similarity(self, x1, x2):
        dl = self.distance(x1, x2)
        logit = self.fc(dl)
        return torch.sigmoid(logit)


# from functional import seq
# original net by Radek
class SiameseNetwork1(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__()
        self.cnn = create_body(arch)
        self.head = nn.Linear(num_features_model(self.cnn), 1)

    def forward(self, im_A, im_B):
        # dl - distance layer
        x1, x2 = seq(im_A, im_B).map(self.cnn).map(self.process_features)
        dl = self.calculate_distance(x1, x2)
        out = self.head(dl)
        return out

    def process_features(self, x):
        return x.reshape(*x.shape[:2], -1).max(-1)[0]

    def calculate_distance(self, x1, x2):
        return (x1 - x2).abs_()


class SiameseNetTriplet(nn.Module):
    def __init__(self, emb_len=128, arch=models.resnet18, width=224, height=224, diff_method=1):
        super().__init__()
        self.cnn = create_body(arch)
        self.fc1 = nn.Linear(cnn_activations_count(arch, width, height), emb_len)
        self.fc2 = nn.Linear(emb_len, 1)
        self.dropout = nn.Dropout(0.0)
        self.diff_method = diff_method
        if self.diff_method == 1:
            self.diff_vec = self.diff_vec1
        else:
            self.diff_vec = self.diff_vec2
        self.forward = self.forward3

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def im2emb(self, im):
        x = self.cnn(im)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

    def diff_vec1(self, emb1, emb2):
        return torch.abs(emb1 - emb2)

    def diff_vec2(self, emb1, emb2):
        return ((emb1 - emb2) ** 2) / (emb1 + emb2)

    def similarity(self, diff):
        x = self.dropout(diff)
        logits = self.fc2(x)
        return torch.sigmoid(logits)

    def distance(self, emb1, emb2):
        return F.pairwise_distance(emb1, emb2)

    #    def emb2sim(self, emb1, emb2):
    #        "embedding to similarity"
    #        dist = self.distance(emb1, emb2)
    #        logits = self.fc2(dist)
    #        similarity = torch.sigmoid(logits)
    #        return similarity
    #
    #    def im2sim(self, im_a, im_b):
    #        "image to similarity"
    #        emb_a = self.im2emb(im_a)
    #        emb_b = self.im2emb(im_b)
    #        return self.emb2sim(emb_a, emb_b)

    # batch hard strategy
    def forward_batch_hard(self, batch):
        embeddings = self.im2emb(batch)
        dist_rows = []
        for emb in embeddings:
            row_emb = emb.expand(embeddings.shape)
            dist = self.distance(row_emb, embeddings)
            dist_rows.append(dist)
        dist_matrix = torch.stack(dist_rows)
        return dist_matrix

    def forward3(self, batch):
        anchor_emb = self.im2emb(batch['anchors'])
        pos_emb = self.im2emb(batch['pos_ims'])
        neg_emb = self.im2emb(batch['neg_ims'])

        pos_dist = self.distance(anchor_emb, pos_emb)
        neg_dist = self.distance(anchor_emb, neg_emb)

        return pos_dist, neg_dist

    def forward2(self, batch):
        anchor_emb = self.im2emb(batch['anchors'])
        pos_emb = self.im2emb(batch['pos_ims'])
        neg_emb = self.im2emb(batch['neg_ims'])

        pos_dist = self.distance(anchor_emb, pos_emb)
        neg_dist = self.distance(anchor_emb, neg_emb)

        pos_diff = self.diff_vec(anchor_emb, pos_emb)
        pos_similarity = self.similarity(pos_diff)
        neg_diff = self.diff_vec(anchor_emb, neg_emb)
        neg_similarity = self.similarity(neg_diff)

        return pos_dist, pos_similarity, neg_dist, neg_similarity

    def forward1(self, batch):
        emb1 = self.im2emb(batch['im1'])
        emb2 = self.im2emb(batch['im2'])
        dist = self.distance(emb1, emb2)
        diff = self.diff_vec(emb1, emb2)
        similarity = self.similarity(diff)
        return dist, similarity


def triplet_loss(dist_matrix, targets, mask_labels=[], margin=0.2):
    # generate pos_mask, todo: mask new_whale from pos-pos
    pos_mask = torch.zeros_like(dist_matrix)
    for j, t1 in enumerate(targets):
        for k, t2 in enumerate(targets):
            if t1 == t2 and t1 not in mask_labels:
                pos_mask[j, k] = 1

    # find pos-max and neg-min
    dist_pos = dist_matrix * pos_mask
    dist_pos_max = dist_pos.sort()[0][:, -1].mean()
    dist_neg = dist_matrix * (1 - pos_mask)
    dist_neg_min = dist_neg.sort()[0][:, 0].mean()

    # triplet loss
    loss = torch.relu(dist_pos_max + margin - dist_neg_min)
    return loss


def stat_distance(dist_matrix, row_labels, col_labels, mask_labels=[]):
    assert len(row_labels) == dist_matrix.shape[0]
    assert len(col_labels) == dist_matrix.shape[1]
    # generate pos_mask. note: new_whale should not be involved in positive distance and should be in mask_labels
    pos_mask = torch.zeros_like(dist_matrix)
    neg_mask = torch.zeros_like(dist_matrix)
    for j, rl in enumerate(row_labels):
        # print(j)
        rl_ex = rl.expand(col_labels.shape)
        if rl not in mask_labels:
            pos_mask[j] = (rl_ex == col_labels)
        neg_mask[j] = (rl_ex != col_labels)

    # find pos-max and neg-min
    pos_dists_max0 = (dist_matrix * pos_mask).sort()[0][:, -1]
    pos_dist_max = pos_dists_max0[pos_dists_max0 != 0].mean()
    del pos_mask

    neg_matrix = dist_matrix * neg_mask
    del dist_matrix
    neg_matrix[neg_matrix == 0] += neg_matrix.max()  # to get min
    neg_dist_min = neg_matrix.sort()[0][:, 0].mean()
    del neg_matrix

    return pos_dist_max, neg_dist_min


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, cut_ratio=4):
        super().__init__()
        self.margin = margin
        self.cut_ratio = cut_ratio

    def forward(self, distances, targets, pos_mask):
        pos_dist = distances[0][pos_mask > 0].sort()[0]
        neg_dist = distances[1].sort()[0]

        neg_dist_min = neg_dist[:max(len(neg_dist) // self.cut_ratio, 8)].mean()
        if len(pos_dist):
            pos_dist_max = pos_dist[-max(len(pos_dist) // self.cut_ratio, 8):].mean()
        else:
            pos_dist_max = neg_dist_min - self.margin

        # triplet loss
        loss = torch.relu(pos_dist_max + self.margin - neg_dist_min)
        return loss


contrastive_neg_margin = 10.0


class ContrastiveLoss2(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=10.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9
        self.reduction = reduction

    def forward(self, distances, targets):
        losses = torch.mean(targets.float() * torch.pow(distances, 2) + (1 - targets).float() * torch.pow(
            torch.relu(self.margin - distances), 2))
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=10.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9
        self.reduction = reduction

    def forward(self, distances, target):
        pp_loss = distances[target == 1]
        pn_dist = distances[target == 0]
        pn_loss = torch.relu(self.margin - pn_dist)
        losses = torch.cat([pp_loss, pn_loss])
        losses = losses[losses != 0]

        # losses = target.float() * distances + (1 - target).float() * torch.relu(self.margin - distances)
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class ContrastiveLoss1(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=10.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.reduction = reduction

    def forward(self, distances, target):
        losses = target.float() * distances + (1 - target).float() * torch.relu(self.margin - distances)
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


def avg_pos_dist(preds, targets):
    pp_loss = preds[targets == 1]
    pp_loss = pp_loss[pp_loss != 0]
    return pp_loss.mean()


def avg_neg_dist(preds, targets):
    pn_loss = preds[targets == 0]
    pn_loss = torch.relu(contrastive_neg_margin - pn_loss)
    pn_loss = pn_loss[pn_loss != 0]
    return pn_loss.mean()


def avg_pos_sim(preds, targets):
    preds = torch.sigmoid(preds.detach())
    pp_loss = preds[targets == 1]
    return pp_loss.mean()


def avg_neg_sim(preds, targets):
    preds = torch.sigmoid(preds.detach())
    pn_loss = preds[targets == 0]
    return pn_loss.mean()


def normalize_batch(batch):
    stat_tensors = [torch.tensor(l).to(device) for l in imagenet_stats]
    return [normalize(batch[0][0], *stat_tensors), normalize(batch[0][1], *stat_tensors)], batch[1]


class CbDists(fastai.callbacks.tracker.TrackerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.pp_loss = 0
        self.pn_loss = 0
        self.momentum = 0.95

    def on_epoch_begin(self, **kwargs):
        self.pp_loss = 0
        self.pn_loss = 0
        self.momentum = 0.95

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.pp_loss = self.momentum * self.pp_loss + (1 - self.momentum) * avg_pos_dist(last_output, last_target)
        self.pn_loss = self.momentum * self.pn_loss + (1 - self.momentum) * avg_neg_dist(last_output, last_target)
        # print(f'pp_dist: {self.pp_dist}, pn_dist: {self.pn_dist}')

    def on_epoch_end(self, **kwargs):
        print(f'pp_loss: {self.pp_loss}, pn_loss: {self.pn_loss}')


class CbSims(fastai.callbacks.tracker.TrackerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.pp_similarity = 1
        self.pn_similarity = 0
        self.momentum = 0.95

    def on_epoch_begin(self, **kwargs):
        # self.pp_similarity = 1
        # self.pn_similarity = 0
        self.momentum = 0.95
        # self.cnt = 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        pos_sim = avg_pos_sim(last_output, last_target)
        if not torch.isnan(pos_sim):
            self.pp_similarity = self.momentum * self.pp_similarity + (1 - self.momentum) * pos_sim

        neg_sim = avg_neg_sim(last_output, last_target)
        if not torch.isnan(neg_sim):
            self.pn_similarity = self.momentum * self.pn_similarity + (1 - self.momentum) * neg_sim

        # self.cnt += 1
        # print(f'{self.cnt}, pp_similarity: {self.pp_similarity}, pn_similarity: {self.pn_similarity}')

    def on_epoch_end(self, **kwargs):
        print(f'pp_similarity: {self.pp_similarity}, pn_similarity: {self.pn_similarity}')
        if self.pn_similarity > 0.5:
            print(f'pn_similarity = {self.pn_similarity}, too high, stop training Coach')
            self.learn.enable_coach = False
        else:
            self.learn.enable_coach = True
        print('#####################################################################################')
        print('\n')


def ds_siamese_emb1(dl, model, ds_with_target=True):
    embs = []
    if ds_with_target:
        targets = []
        for k, (data, target) in enumerate(dl):
            # print(k)
            embs.append(model.im2emb(data))
            targets.append(target)
        return embs, targets
    else:
        for data in dl:
            embs.append(model.im2emb(data))
        return embs


def ds_siamese_emb(dl, model, ds_with_target=True):
    embs = []
    if ds_with_target:
        targets = []
        for k, (data, target) in enumerate(dl):
            if k % 10 == 0:
                print(k)
            # data = data.to(device)
            embs.append(model.im2emb(data))
            targets.append(target)
        return embs, targets
    else:
        for data in dl:
            embs.append(model.im2emb(data))
        return embs


def cal_mat(model, data_loader1, data_loader2, ds_with_target1=True, ds_with_target2=True):
    mat_file = 'matrix.dump'
    if os.path.isfile(mat_file):
        with open(mat_file, 'rb') as f:
            return pickle.load(f)
    model.eval()
    with torch.no_grad():
        target1_tensor = None
        target2_tensor = None
        # calculate embeddings of all validation_set
        if ds_with_target1:
            emb1, target1 = ds_siamese_emb(data_loader1, model, ds_with_target=ds_with_target1)
            target1_tensor = torch.cat(target1)
        else:
            emb1 = ds_siamese_emb(data_loader1, model, ds_with_target=ds_with_target1)
        emb1_tensor = torch.cat(emb1)

        if ds_with_target2:
            emb2, target2 = ds_siamese_emb(data_loader2, model, ds_with_target=ds_with_target2)
            target2_tensor = torch.cat(target2)
        else:
            emb2 = ds_siamese_emb(data_loader2, model, ds_with_target=ds_with_target2)
        emb2_tensor = torch.cat(emb2)

        # calculate distances between all emb1 and all emb2
        distances = []
        print(f'{now2str()} calculate distance matrix')
        for i, emb1 in enumerate(emb1_tensor):
            print(f'{now2str()} calculate row {i}')
            emb1_ex = emb1.expand(emb2_tensor.shape)
            dists = model.distance(emb1_ex, emb2_tensor)
            # dists = model.similarity(emb1_ex, emb2_tensor).view(-1)
            distances.append(dists)
        distance_matrix = torch.stack(distances)
        with open(mat_file, 'wb') as f:
            pickle.dump((distance_matrix, target1_tensor, target2_tensor), f)
        print('over')
        return distance_matrix, target1_tensor, target2_tensor


def siamese_mat(in_dl, model, rf_dl, pos_mask=[], ref_idx2class=[], target_idx2class=[], enable_cal_dist=True):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all validation_set
        val_embs, targets = ds_siamese_emb(in_dl, model, ds_with_target=True)
        val_emb_tensor = torch.cat(val_embs)
        val_target_tensor = torch.cat(targets)
        # print(val_emb_tensor.shape, val_target_tensor.shape)

        # calculate embeddings of all classes except new_whale
        rf_embs, rf_targets = ds_siamese_emb(rf_dl, model, ds_with_target=True)
        rf_emb_tensor = torch.cat(rf_embs)
        rf_target_tensor = torch.cat(rf_targets).to(device)

        # calculate distances between all val_embs and all rf_embs
        distances = []
        # print(f'{now2str()} calculate distance matrix')
        for i, val_emb in enumerate(val_emb_tensor):
            # print(f'{now2str()} calculate row {i}')
            distance_list = []
            val_emb_batch = val_emb.expand(rf_emb_tensor.shape)
            dists = model.diff(val_emb_batch, rf_emb_tensor)
            # dists = model.similarity(val_emb_batch, rf_emb_tensor).view(-1)
            distances.append(dists)
        distance_matrix = torch.stack(distances)

        # find average max positive distance and average min negative distance
        # print(f'{now2str()} find max positive distance and min negative distance')
        dist_pos_max, dist_neg_min = -1, -1
        if enable_cal_dist:
            dist_pos_max, dist_neg_min = stat_distance(distance_matrix, val_target_tensor, rf_target_tensor,
                                                       mask_labels=pos_mask)
            print(f'dist_pos_max = {dist_pos_max}, dist_neg_min = {dist_neg_min}')

        # todo:insert new_whale

        # cal map5
        top5_matrix, map5 = cal_mapk(distance_matrix, val_target_tensor, k=5, ref_idx2class=ref_idx2class,
                                     target_idx2class=target_idx2class)
        print(f'map5 = {map5}')
        torch.cuda.empty_cache()
        return map5, top5_matrix, dist_pos_max, dist_neg_min


class SiameseValidateCallback(fastai.callbacks.tracker.TrackerCallback):
    "A `Callback` to validate SiameseNet."

    def __init__(self, learn, txlog=None):
        super().__init__(learn)
        self.txlog = txlog

    # def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        "Stop the training if necessary."
        print(f'Epoch {epoch} validation:')
        new_whale_idx = find_new_whale_idx(self.learn.data.train_dl.ds.y.classes)
        map5, top5_matrix, pos_dist_max, neg_dist_min = siamese_mat(self.learn.data.valid_dl, self.learn.model,
                                                                    self.learn.data.fix_dl,
                                                                    pos_mask=[new_whale_idx],
                                                                    ref_idx2class=self.learn.data.fix_dl.ds.y,
                                                                    target_idx2class=self.learn.data.valid_dl.ds.y)
        self.txlog.add_scalar('map5', map5, epoch)
        self.txlog.add_scalar('pos_dist_max', pos_dist_max, epoch)
        self.txlog.add_scalar('neg_dist_min', neg_dist_min, epoch)
        self.txlog.add_scalar('dist_diff', neg_dist_min - pos_dist_max, epoch)
        print('\n')


@dataclass
class LearnerEx(Learner):
    enable_validate: bool = True

    def __post_init__(self):
        super().__post_init__()

    def fit(self, epochs: int, lr: Union[Floats, slice] = defaults.lr,
            wd: Floats = None, callbacks: Collection[Callback] = None) -> None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        if not getattr(self, 'opt', False):
            self.create_opt(lr, wd)
        else:
            self.opt.lr, self.opt.wd = lr, wd
        callbacks = [cb(self) for cb in self.callback_fns] + listify(callbacks)
        fit(epochs, self.model, self.loss_func, opt=self.opt, data=self.data, metrics=self.metrics,
            callbacks=self.callbacks + callbacks, enable_validate=self.enable_validate)


def fit(epochs: int, model: nn.Module, loss_func: LossFunction, opt: optim.Optimizer,
        data: DataBunch, callbacks: Optional[CallbackList] = None, metrics: OptMetrics = None,
        enable_validate=True) -> None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
    cb_handler = CallbackHandler(callbacks, metrics)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception = False
    try:
        for epoch in pbar:
            model.train()
            cb_handler.on_epoch_begin()

            for xb, yb in progress_bar(data.train_dl, parent=pbar):
                xb = torch.cat(xb)
                yb = torch.cat(yb)
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(model, xb, yb, loss_func, opt, cb_handler)
                if cb_handler.on_batch_end(loss): break

            if enable_validate and not data.empty_val:
                val_loss = validate(model, data.valid_dl, loss_func=loss_func,
                                    cb_handler=cb_handler, pbar=pbar)
            else:
                val_loss = None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise e
    finally:
        cb_handler.on_train_end(exception)


def gen_class_reference(ds):
    class_1_dict = make_class_1_idx_dict(ds, ignore=['new_whale'])
    return class_1_dict


def gen_submission(top5_matrix, image_files, out_file):
    top5_classes = []
    for top5 in top5_matrix:
        top5_classes.append(' '.join([t for t in top5]))

    submission = pd.DataFrame({'Image': [path.name for path in image_files]})
    submission['Id'] = top5_classes
    # submission.to_csv(f'../submission/{out_file}.csv.gz', index=False, compression='gzip')
    submission.to_csv(f'../submission/{out_file}.csv', index=False)


def intersection(preds, targs):
    # preds and targs are of shape (bs, 4), pascal_voc format
    max_xy = torch.min(preds[:, 2:], targs[:, 2:])
    min_xy = torch.max(preds[:, :2], targs[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def area(boxes):
    return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))


def union(preds, targs):
    return area(preds) + area(targs) - intersection(preds, targs)


def IoU(preds, targs):
    return intersection(preds, targs) / union(preds, targs)


class Metric(nn.Module):
    def __init__(self, emb_sz=64):
        super().__init__()
        self.l = nn.Linear(emb_sz * 2, emb_sz * 2, False)

    def forward(self, d):
        d2 = d.pow(2)
        d = self.l(torch.cat((d, d2), dim=-1))
        x = d.pow(2).sum(dim=-1)
        return x.view(-1)


class Contrastive_loss(nn.Module):
    def __init__(self, m=10.0, wd=1e-4):
        super().__init__()
        self.m, self.wd = m, wd

    def forward(self, d, target):
        d = d.float()
        # matrix of all vs all comparisons
        t = torch.cat(target)
        sz = t.shape[0]
        t1 = t.unsqueeze(1).expand((sz, sz))
        t2 = t1.transpose(0, 1)
        y = ((t1 == t2) + to_gpu(torch.eye(sz).byte())).view(-1)

        loss_p = d[y == 1]
        loss_n = F.relu(self.m - torch.sqrt(d[y == 0])) ** 2
        loss = torch.cat((loss_p, loss_n), 0)
        loss = loss[torch.nonzero(loss).squeeze()]
        loss = loss.mean() if loss.shape[0] > 0 else loss.sum()
        loss += self.wd * (d ** 2).mean()  # compactification term
        return loss


# accuracy within a triplet
def T_acc(d, target):
    sz = target[0].shape[0]
    lp = [3 * sz * i + i + sz for i in range(sz)]
    ln = [3 * sz * i + i + 2 * sz for i in range(sz)]
    dp, dn = d[lp], d[ln]
    return (dp < dn).float().mean()


# accuracy within a hardest triplet in a batch for each anchor image
def BH_acc(d, target):
    t = torch.cat(target)
    sz = t.shape[0]
    t1 = t.unsqueeze(1).expand((sz, sz))
    t2 = t1.transpose(0, 1)
    y = (t1 == t2)
    d = d.float().view(sz, sz)
    BH = []
    for i in range(sz):
        dp = d[i, y[i, :] == 1].max()
        dn = d[i, y[i, :] == 0].min()
        BH.append(dp < dn)
    return torch.FloatTensor(BH).float().mean()


def pp_dist_max(d, target):
    t = torch.cat(target)
    sz = t.shape[0]
    t1 = t.unsqueeze(1).expand((sz, sz))
    t2 = t1.transpose(0, 1)
    y = (t1 == t2)
    d = d.float().view(sz, sz)
    pp_dist = []
    for i in range(sz):
        dp = d[i, y[i] == 1].max()
        dn = d[i, y[i] == 0].min()
        pp_dist.append(dp)
    return torch.FloatTensor(pp_dist).float().mean()


def pn_dist_min(d, target):
    t = torch.cat(target)
    sz = t.shape[0]
    t1 = t.unsqueeze(1).expand((sz, sz))
    t2 = t1.transpose(0, 1)
    y = (t1 == t2)
    d = d.float().view(sz, sz)
    pn_dist = []
    for i in range(sz):
        dn = d[i, y[i] == 0].min()
        pn_dist.append(dn)
    return torch.FloatTensor(pn_dist).float().mean()


def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]


def get_densenet121(pre=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = cut_model(cut_model(densenet121(pre), -1)[0], -1)
    return nn.Sequential(*layers)


class TripletDenseNet121(nn.Module):
    def __init__(self, pre=True, emb_sz=64, ps=0.5):
        super().__init__()
        encoder = get_densenet121(pre)
        # add DataParallel to allow support of multiple GPUs
        self.cnn = nn.DataParallel(nn.Sequential(encoder[0], encoder[1], nn.ReLU(),
                                                 encoder[3], encoder[4], encoder[5], encoder[6], encoder[7],
                                                 encoder[8], encoder[9], encoder[10]))
        self.head = nn.DataParallel(nn.Sequential(AdaptiveConcatPool2d(), Flatten(),
                                                  nn.BatchNorm1d(2048), nn.Dropout(ps), nn.Linear(2048, 1024),
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(1024), nn.Dropout(ps), nn.Linear(1024, emb_sz)))
        self.metric = nn.DataParallel(Metric(emb_sz))

    def forward(self, x):
        x1, x2, x3 = x[:, 0, :, :, :], x[:, 1, :, :, :], x[:, 2, :, :, :]
        x1 = self.head(self.cnn(x1))
        x2 = self.head(self.cnn(x2))
        x3 = self.head(self.cnn(x3))
        x = torch.cat((x1, x2, x3))
        sz = x.shape[0]
        x1 = x.unsqueeze(1).expand((sz, sz, -1))
        x2 = x1.transpose(0, 1)
        # matrix of all vs all differencies
        d = (x1 - x2).view(sz * sz, -1)
        return self.metric(d)

    def get_embedding(self, x):
        return self.head(self.cnn(x))

    def get_d(self, x0, x):
        d = (x - x0)
        return self.metric(d)


class DenseNet121Model():
    def __init__(self, pre=True, name='TripletDenseNet21', **kwargs):
        self.model = to_gpu(TripletDenseNet121(pre=True, **kwargs))
        self.name = name

    def get_layer_groups(self, precompute):
        m = self.model.module if isinstance(self.model, FP16) else self.model
        if precompute:
            return [m.head] + [m.metric]
        c = children(m.cnn.module)
        return list(split_by_idxs(c, [8])) + [m.head] + [m.metric]
