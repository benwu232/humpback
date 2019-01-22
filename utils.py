import numpy as np
import torch
import pandas as pd
import torch
from copy import deepcopy
import datetime as dt

import fastai
from fastai.vision import *
from fastai.basic_data import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks.hooks import num_features_model, model_sizes
import torchvision
import tensorboardX as tx

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

train_list_dbg = ['00050a15a.jpg', '0005c1ef8.jpg', '0006e997e.jpg', '833675975.jpg', '2fe2cc5c0.jpg', '2f31725c6.jpg', '30eac8c9f.jpg', '3c4235ad2.jpg', '4e6290672.jpg', 'b2cabd9d8.jpg', '411ae328a.jpg', '204c7a64b.jpg', '3161ae9b9.jpg', '399533efd.jpg', '108f230d8.jpg', '2c63ff756.jpg', '1eccb4eba.jpg', '20e7c6af4.jpg', '23acb3957.jpg', '0d5777fc2.jpg', '2885ea466.jpg', '2d43e205a.jpg', '2c0892b4d.jpg', '847b16277.jpg', '03be3723c.jpg', '1cc6523fc.jpg', '47d9ad983.jpg', '645300669.jpg', '77fef7f1a.jpg', '67947e46b.jpg', '4bb9c9727.jpg', '166a9e05d.jpg', '2b95bba09.jpg', '6662f0d1c.jpg']
val_list_dbg = ['ffd61cded.jpg', 'ffdddcc0f.jpg', 'fffde072b.jpg', 'c5658abf0.jpg', 'cc6c1a235.jpg', 'ee87a2369.jpg', '8d601c3e1.jpg', 'b758f5366.jpg', 'b9aefbbc8.jpg', 'c5d1963ab.jpg', '910e6297a.jpg', 'bdb41dba8.jpg', 'bf1eba5c2.jpg', 'cd650e905.jpg', 'b3ba738d7.jpg', 'c48bb3cbf.jpg', 'c5f7b85de.jpg', '79fac8c6d.jpg', 'b94450f8e.jpg', 'bd3f7ba02.jpg', 'c938f9d6f.jpg', 'e92c45748.jpg', '9ae676cb0.jpg', 'a8d5237c0.jpg']

def now2str(format="%Y-%m-%d_%H:%M:%S"):
    #str_time = time.strftime("%Y-%b-%d-%H-%M-%S", time.localtime(time.time()))
    return dt.datetime.now().strftime(format)

def find_new_whale_idx(classes):
    for k, cls in enumerate(classes):
        if cls == 'new_whale':
            return k

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

def cal_mapk(data_matrix:tensor, targets:tensor, target_idx2class=[], k=5, average=True, threshold=2.0, ref_idx2class=[], descending=True):
    topk_dists, topk_idxes = data_matrix.sort(dim=1, descending=descending)
    topk_dists = topk_dists.detach().cpu().numpy()
    topk_idxes = topk_idxes.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    #topk_matrix = 0 - np.ones((data_matrix.shape[0], k), dtype=np.int32)
    topk_matrix = [['' for _ in range(topk_idxes.shape[1])] for _ in range(topk_idxes.shape[0])]
    mapk = np.zeros(len(targets))
    for row, (values, idxes) in enumerate(zip(topk_dists, topk_idxes)):
        c = 0
        for col, (value, idx) in enumerate(zip(values, idxes)):
            if value > threshold and 'new_whale' not in topk_matrix[row]:
                topk_matrix[row][c] = 'new_whale'
                c += 1
                if target_idx2class[targets[row]].obj == 'new_whale' and mapk[row] == 0:
                    mapk[row] = 1 / c
            else:
                class_str = ref_idx2class[idx].obj
                if class_str not in topk_matrix[row]:
                    topk_matrix[row][c] = class_str
                    c += 1
                    if target_idx2class[targets[row]].obj == class_str and mapk[row] == 0:
                        mapk[row] = 1 / c

            if c >= k:
                break
    if average:
        mapk = mapk.mean()

    return topk_matrix, mapk


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


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def map5(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    res = mapk([[t] for t in targs.cpu().numpy()], top_5.cpu().numpy(), 5)
    return torch.tensor(res)


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


def split_whale_set(df, nth_fold=0, total_folds=5, new_whale_method=0, seed=1):
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
    # df_known = df[df.Id!='new_whale']
    for name, group in df.groupby('Id'):
        # print(name, len(group), group.index, type(group))
        # if name == 'w_b82d0eb':
        #    print(name, df_known[df_known.Id==name])
        if new_whale_method == 0 and name == 'new_whale':
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


def make_whale_class_dict(df):
    whale_class_dict = {}
    for name, group in df.groupby('Id'):
        whale_class_dict[name] = group.Image.tolist()
    return whale_class_dict


class ImageItemListEx(ImageItemList):
    def __init__(self, *args, convert_mode='RGB', **kwargs):
        super().__init__(*args, convert_mode=convert_mode, **kwargs)

    def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):
        "Set `tfms` to be applied to the xs of the train and validation set."
        if not tfms: return self
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self


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
        #if len(sample) == 2:
            return sample[0], sample[1]
        else:
            return sample


class Class1Dataset(SimpleDataset):
    def __init__(self, ds, class_dict):
        super().__init__(ds)
        self.class_dict = class_dict
        self.keys = sorted(list(self.class_dict.keys()))

    def __getitem__(self, index):
        key = self.keys[index]
        real_idx = self.class_dict[key]
        sample = self.ds[real_idx].x[0]
        return sample


class DataLoaderTrain1(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()
        #self.mean, self.std = torch.tensor(imagenet_stats)
        self.stats = tensor(imagenet_stats).to(device)

    def proc_batch(self, b):
        data = {}
        for name in b[0]:
            data[name] = []
            for im in b[0][name]:
                #im.show()
                transformed = im.apply_tfms(listify(self.tfms))
                #transformed.show()
                normalized = torchvision.transforms.functional.normalize(transformed.data, self.stats[0], self.stats[1])
                data[name].append(normalized)
            data[name] = torch.stack(data[name]).to(device)
        target = b[1].to(device)
        return data, target


class DataLoaderTrain(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()
        #self.mean, self.std = torch.tensor(imagenet_stats)
        self.stats = tensor(imagenet_stats).to(device)

    def proc_batch(self, b):
        data = {}
        for name in b[0]:
            if name == 'pos_valid_masks':
                data[name] = torch.tensor(b[0][name]).to(device)
                continue
            data[name] = []
            for im in b[0][name]:
                #im.show()
                transformed = im.apply_tfms(listify(self.tfms))
                #transformed.show()
                data[name].append(transformed.data)

            #data[name] = normalize(torch.stack(data[name]), self.mean, self.std).to(device)
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
    for k, y in enumerate(ds.y):
        if y.obj in ignore:
            continue
        if y.data not in class_idx_dict:
            class_idx_dict[y.data] = [k]
        else:
            class_idx_dict[y.data].append(k)

    for idx in class_idx_dict:
        class_idx_dict[idx] = np.asarray(class_idx_dict[idx])

    return class_idx_dict

class SiameseDs(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.idx2class = ds.y.items
        #self.id_dict = make_id_dict(self.whale_class_dict)
        self.class_dict = make_class_idx_dict(ds)
        self.len = len(self.ds)
        self.new_whale = find_new_whale_idx(ds.y.classes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if np.random.choice([0, 1]):
            while True:
                idx_p, pos_valid = self.find_same(idx)
                if pos_valid:
                    return (self.ds[idx][0], self.ds[idx_p][0]), 1
                else:
                    idx = np.random.randint(self.len)
        else:
            idx_n = self.find_different(idx)
            return (self.ds[idx][0], self.ds[idx_n][0]), 0

    def find_same(self, idx):
        whale_class = self.idx2class[idx]
        class_members = self.class_dict[whale_class]

        # for 1-class or new_whale, there is no same
        if len(class_members) == 1 or whale_class == self.new_whale:
            return idx, 0  # 0 means anchor-positive pair is not valid

        candidates = class_members[class_members != idx] #same label but not same image
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


class SiameseDsTriplet(SiameseDs):
    def __init__(self, ds):
        self.ds = ds
        self.idx2class = ds.y.items
        #self.id_dict = make_id_dict(self.whale_class_dict)
        self.class_dict = make_class_idx_dict(ds)
        self.len = len(self.ds)
        self.new_whale = find_new_whale_idx(ds.y.classes)

    def __getitem__(self, idx):
        idx_a = idx
        idx_p, pos_valid = self.find_same(idx)
        idx_n = self.find_different(idx)
        return self.ds[idx_a][0], self.ds[idx_p][0], self.ds[idx_n][0], pos_valid


def collate_siamese(items):
    im1_list = []
    im2_list = []
    target_list = []

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
    #batch['pos_valid_masks'] = np.stack(pos_valid_masks)
    batch_len = len(batch['anchors'])
    target = torch.cat((torch.ones(batch_len, dtype=torch.int32), torch.zeros(batch_len, dtype=torch.int32)))
    pos_mask = torch.tensor(pos_valid_masks, dtype=torch.int32)
    return batch, (target, pos_mask)


def cnn_activations_count(model, width, height):
    _, ch, h, w = model_sizes(create_body(model), (width, height))[-1]
    return ch * h * w


class SiameseNet(nn.Module):
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

    def distance(self, emb1, emb2):
        return F.pairwise_distance(emb1, emb2)

    def similarity(self, dist):
        x = self.dropout(dist)
        logits = self.fc2(x)
        return torch.sigmoid(logits)

    def forward(self, batch):
        emb1 = self.im2emb(batch['im1'])
        emb2 = self.im2emb(batch['im2'])
        diff = self.diff_vec1(emb1, emb2)
        similarity = self.similarity(diff)
        return similarity


from functional import seq
#original net by Radek
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

#modified Radek's net not using seq
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

#for contrastive loss
class SiameseNetwork2(nn.Module):
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
        dist = self.distance(x1, x2)
        return dist

    def process_features(self, x):
        return x.reshape(*x.shape[:2], -1).max(-1)[0]

    def distance(self, x1, x2):
        return (x1 - x2).abs().mean(dim=1)


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

    #batch hard strategy
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
    #generate pos_mask, todo: mask new_whale from pos-pos
    pos_mask = torch.zeros_like(dist_matrix)
    for j, t1 in enumerate(targets):
        for k, t2 in enumerate(targets):
            if t1 == t2 and t1 not in mask_labels:
                pos_mask[j, k] = 1

    #find pos-max and neg-min
    dist_pos = dist_matrix * pos_mask
    dist_pos_max = dist_pos.sort()[0][:, -1].mean()
    dist_neg = dist_matrix * (1 - pos_mask)
    dist_neg_min = dist_neg.sort()[0][:, 0].mean()

    #triplet loss
    loss = torch.relu(dist_pos_max + margin - dist_neg_min)
    return loss

def stat_distance(dist_matrix, row_labels, col_labels, mask_labels=[]):
    assert len(row_labels) == dist_matrix.shape[0]
    assert len(col_labels) == dist_matrix.shape[1]
    #generate pos_mask. note: new_whale should not be involved in positive distance and should be in mask_labels
    pos_mask = torch.zeros_like(dist_matrix)
    neg_mask = torch.zeros_like(dist_matrix)
    for j, rl in enumerate(row_labels):
        #print(j)
        rl_ex = rl.expand(col_labels.shape)
        if rl not in mask_labels:
            pos_mask[j] = (rl_ex == col_labels)
        neg_mask[j] = (rl_ex != col_labels)

    #find pos-max and neg-min
    pos_dists_max0 = (dist_matrix * pos_mask).sort()[0][:, -1]
    pos_dist_max = pos_dists_max0[pos_dists_max0 != 0].mean()

    neg_matrix = dist_matrix * neg_mask
    neg_matrix[neg_matrix == 0] += neg_matrix.max() #to get min
    neg_dist_min = neg_matrix.sort()[0][:, 0].mean()

    return pos_dist_max, neg_dist_min


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, cut_ratio=4):
        super().__init__()
        self.margin = margin
        self.cut_ratio = cut_ratio

    def forward(self, distances, targets, pos_mask):
        pos_dist = distances[0][pos_mask > 0].sort()[0]
        neg_dist = distances[1].sort()[0]

        neg_dist_min = neg_dist[:max(len(neg_dist)//self.cut_ratio, 8)].mean()
        if len(pos_dist):
            pos_dist_max = pos_dist[-max(len(pos_dist)//self.cut_ratio, 8):].mean()
        else:
            pos_dist_max = neg_dist_min - self.margin

        #triplet loss
        loss = torch.relu(pos_dist_max + self.margin - neg_dist_min)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, distances, target, size_average=True):
        losses = target.float() * distances + (1 - target).float() * torch.relu(self.margin - distances)
        return losses.mean() if size_average else losses.sum()


def normalize_batch(batch):
    stat_tensors = [torch.tensor(l).to(device) for l in imagenet_stats]
    return [normalize(batch[0][0], *stat_tensors), normalize(batch[0][1], *stat_tensors)], batch[1]


def ds_siamese_emb1(dl, model, ds_with_target=True):
    embs = []
    if ds_with_target:
        targets = []
        for k, (data, target) in enumerate(dl):
            #print(k)
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
            #print(k)
            embs.append(model.im2emb(data))
            targets.append(target)
        return embs, targets
    else:
        for data in dl:
            embs.append(model.im2emb(data))
        return embs


def siamese_validate(val_dl, model, rf_dl, pos_mask=[], ref_idx2class=[], target_idx2class=[]):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all validation_set
        val_embs, targets = ds_siamese_emb(val_dl, model, ds_with_target=True)
        val_emb_tensor = torch.cat(val_embs)
        val_target_tensor = torch.cat(targets)
        #print(val_emb_tensor.shape, val_target_tensor.shape)

        # calculate embeddings of all classes except new_whale
        rf_embs, rf_targets = ds_siamese_emb(rf_dl, model, ds_with_target=True)
        rf_emb_tensor = torch.cat(rf_embs)
        rf_target_tensor = torch.cat(rf_targets)

        # calculate distances between all val_embs and all rf_embs
        distances = []
        #print(f'{now2str()} calculate distance matrix')
        for i, val_emb in enumerate(val_emb_tensor):
            #print(f'{now2str()} calculate row {i}')
            distance_list = []
            val_emb_batch = val_emb.expand(rf_emb_tensor.shape)
            #dists = model.distance(val_emb_batch, rf_emb_tensor)
            dists = model.similarity(val_emb_batch, rf_emb_tensor).view(-1)
            distances.append(dists)
        distance_matrix = torch.stack(distances)

        #find average max positive distance and average min negative distance
        #print(f'{now2str()} find max positive distance and min negative distance')
        dist_pos_max, dist_neg_min = stat_distance(distance_matrix, val_target_tensor, rf_target_tensor, mask_labels=pos_mask)
        print(f'dist_pos_max = {dist_pos_max}, dist_neg_min = {dist_neg_min}')

        # todo:insert new_whale

        # cal map5
        top5_matrix, map5 = cal_mapk(distance_matrix, val_target_tensor, k=5, ref_idx2class=ref_idx2class, target_idx2class=target_idx2class)
        print(f'map5 = {map5}')
        return map5, dist_pos_max, dist_neg_min

class SiameseValidateCallback(fastai.callbacks.tracker.TrackerCallback):
    "A `Callback` to validate SiameseNet."

    def __init__(self, learn, txlog=None):
        super().__init__(learn)
        self.txlog = txlog

    #def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        "Stop the training if necessary."
        print(f'Epoch {epoch} validation:')
        new_whale_idx = find_new_whale_idx(self.learn.data.train_dl.ds.y.classes)
        map5, pos_dist_max, neg_dist_min = siamese_validate(self.learn.data.valid_dl, self.learn.model, self.learn.data.fix_dl,
                                                            pos_mask=[new_whale_idx], ref_idx2class=self.learn.data.fix_dl.ds.y,
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

