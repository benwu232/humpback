import numpy as np
import torch
import pandas as pd
import torch

import fastai
from fastai.vision import *
from fastai.basic_data import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks.hooks import num_features_model, model_sizes

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

train_list_dbg = ['00050a15a.jpg', '833675975.jpg', '2fe2cc5c0.jpg', '2f31725c6.jpg', '4e6290672.jpg', 'b2cabd9d8.jpg', '411ae328a.jpg', '204c7a64b.jpg', '108f230d8.jpg', '2c63ff756.jpg', '1eccb4eba.jpg', '0d5777fc2.jpg', '2c0892b4d.jpg', '847b16277.jpg', '03be3723c.jpg', '1cc6523fc.jpg', '77fef7f1a.jpg', '67947e46b.jpg', '4bb9c9727.jpg', '166a9e05d.jpg', '6662f0d1c.jpg']
val_list_dbg = ['fffde072b.jpg', 'ee87a2369.jpg', '8d601c3e1.jpg', 'c5d1963ab.jpg', '910e6297a.jpg', 'cd650e905.jpg', 'c5f7b85de.jpg', '79fac8c6d.jpg', 'c938f9d6f.jpg', 'e92c45748.jpg', 'a8d5237c0.jpg']

def plot_lr(self):
        if not in_ipynb():
            plt.switch_backend('agg')
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.iterations, self.lrs)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))

def cal_mapk(pred_matrix:tensor, targets:tensor, k=5, average=True):
    topk_probs, topk_idxes = pred_matrix.topk(k, dim=1)
    onehots = (topk_idxes == targets.view(-1, 1))
    onehots = onehots.detach().cpu().numpy()

    mapk = np.zeros(len(onehots))
    for i, onehot in enumerate(onehots):
        r = np.where(onehot == 1)[0]
        if r:
            mapk[i] = 1 / (r[0] + 1)
    if average:
        mapk = mapk.mean()
    return mapk


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


def split_whale_set1(df, nth_split=0, total_split=5, use_new_whale=True, seed=1):
    np.random.seed(seed)
    # list(df_known.groupby('Id'))
    train_list = []
    val_list = []
    df_known = df[df.Id != 'new_whale']
    for name, group in df_known.groupby('Id'):
        # print(name, len(group), group.index, type(group))
        # if name == 'w_b82d0eb':
        #    print(name, df_known[df_known.Id==name])
        group_num = len(group)
        idxes = group.index.values
        if group_num > 1:
            np.random.shuffle(idxes)
            # idxes = list(idxes)
            span = max(1, group_num // total_split)
            val_idxes = idxes[nth_split * span:(nth_split + 1) * span]
            train_idxes = list(set(idxes) - set(val_idxes))
            val_list.extend(val_idxes)
            train_list.extend(train_idxes)
        else:
            train_list.extend(idxes)

    if use_new_whale:
        df_new = df[df.Id == 'new_whale']
        train_list.extend(df_new.index.values)

    return train_list, val_list


def make_whale_class_dict(df):
    whale_class_dict = {}
    for name, group in df.groupby('Id'):
        whale_class_dict[name] = group.Image.tolist()
    return whale_class_dict


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



class DataLoaderTrain(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()

    def proc_batch(self, b):
        data = {}
        for name in b[0]:
            data[name] = b[0][name].to(device)
        target = b[1].to(device)
        return data, target


class DataLoaderVal(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()

    def proc_batch(self, b):
        if len(b) == 2:
            data = b[0].to(device)
            target = b[1].to(device)
            return data, target
        else:
            return b.to(device)

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

class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.whale_class_dict = ds.y.items
        #self.id_dict = make_id_dict(self.whale_class_dict)
        self.class_idx_dict = make_class_idx_dict(ds)
        self.len = len(self.ds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx_a = idx
        idx_p, type = self.find_same(idx)
        idx_n = self.find_different(idx)
        return self.ds[idx_a][0], self.ds[idx_p][0], self.ds[idx_n][0], type

    def find_same(self, idx):
        whale_class = self.whale_class_dict[idx]
        c = self.class_idx_dict[whale_class]

        # for 1-class or new_whale, there is no same
        if len(c) == 1 or len(c) > 100:
            return idx, 0  # 0 means no anchor-positive pair

        candidates = c[c != idx]
        np.random.shuffle(candidates)
        return candidates[0], 1

    def find_different(self, idx1, idx2=None):
        # already have idx2
        if idx2 is not None:
            return idx2

        whale_id = self.whale_class_dict[idx1]
        while True:
            idx2 = np.random.randint(self.len // 2)
            if self.whale_class_dict[idx2] != whale_id:
                break
        return idx2


def siamese_collate(items):
    anchor_list = []
    pos_list = []
    neg_list = []
    pos_valid_masks = []

    for anchor, pos, neg, mask in items:
        anchor_list.append(anchor.data)
        pos_list.append(pos.data)
        neg_list.append(neg.data)
        pos_valid_masks.append(mask)

    # batch = []
    # batch.append(torch.tensor(np.stack(anchor_list)).to(device))
    # batch.append(torch.tensor(np.stack(pos_list)).to(device))
    # batch.append(torch.tensor(np.stack(neg_list)).to(device))
    # batch.append(torch.tensor(np.stack(pos_valid_masks)).to(device))
    # return batch, batch[-1] # just for (data, target) format

    batch = {}
    batch['anchors'] = torch.tensor(np.stack(anchor_list))
    batch['pos_ims'] = torch.tensor(np.stack(pos_list))
    batch['neg_ims'] = torch.tensor(np.stack(neg_list))
    batch['pos_valid_masks'] = torch.tensor(np.stack(pos_valid_masks))
    batch_len = len(batch['anchors'])
    target = torch.cat((torch.ones(batch_len, dtype=torch.int32), torch.zeros(batch_len, dtype=torch.int32)))
    return batch, target


def cnn_activations_count(model, width, height):
    _, ch, h, w = model_sizes(create_body(model), (width, height))[-1]
    return ch * h * w


class SiameseNet(nn.Module):
    def __init__(self, emb_len=128, arch=models.resnet18, width=224, height=224, norm=1):
        super().__init__()
        self.cnn = create_body(arch)
        self.fc1 = nn.Linear(cnn_activations_count(arch, width, height), emb_len)
        self.fc2 = nn.Linear(emb_len, 1)
        self.dist_type = norm

    def cal_embedding(self, im):
        x = self.cnn(im)
        x = self.process_features(x)
        x = self.fc1(x)
        emb = torch.sigmoid(x)
        return emb

    def cal_distance(self, emb1, emb2):
        if self.dist_type == 1:
            return torch.abs(emb1 - emb2)
        else:
            return (emb1 - emb2) ** 2

    def classify(self, dist):
        logits = self.fc2(dist)
        return torch.sigmoid(logits)

    def cal_similarity(self, emb1, emb2):
        dist = self.cal_distance(emb1, emb2)
        prob = self.classify(dist)
        return prob

    def forward_test(self, im_a, im_b):
        emb_a = self.cal_embedding(im_a)
        emb_b = self.cal_embedding(im_b)
        dist_v = self.cal_distance(emb_a, emb_b)
        prob = torch.sigmoid(self.fc2(dist_v))
        return prob

    def forward_train(self, batch):
        anchor_emb = self.cal_embedding(batch['anchors'])
        pos_emb = self.cal_embedding(batch['pos_ims'])
        neg_emb = self.cal_embedding(batch['neg_ims'])

        pos_dist = self.cal_distance(anchor_emb, pos_emb)
        neg_dist = self.cal_distance(anchor_emb, neg_emb)

        pos_logits = self.fc2(pos_dist)
        neg_logits = self.fc2(neg_dist)
        # return pos_logits, neg_logits, pos_dist, neg_dist
        return torch.cat((pos_logits, neg_logits))

    def forward(self, batch):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_test(batch)

    def process_features(self, x):
        return x.reshape(x.shape[0], -1)


def normalize_batch(batch):
    stat_tensors = [torch.tensor(l).to(device) for l in imagenet_stats]
    return [normalize(batch[0][0], *stat_tensors), normalize(batch[0][1], *stat_tensors)], batch[1]


def ds_siamese_emb(dl, model, ds_with_target=True):
    embs = []
    if ds_with_target:
        targets = []
        for k, (data, target) in enumerate(dl):
            #print(k)
            embs.append(model.cal_embedding(data))
            targets.append(target)
        return embs, targets
    else:
        for data in dl:
            embs.append(model.cal_embedding(data))
        return embs


def siamese_validate(val_dl, model, train_rf_dl):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all validation_set
        val_embs, targets = ds_siamese_emb(val_dl, model, ds_with_target=True)
        val_emb_tensor = torch.cat(val_embs)
        val_target_tensor = torch.cat(targets)
        #print(val_emb_tensor.shape, val_target_tensor.shape)

        # calculate embeddings of all classes except new_whale
        train_rf_embs, rf_targets = ds_siamese_emb(train_rf_dl, model, ds_with_target=True)
        train_rf_emb_tensor = torch.cat(train_rf_embs)
        rf_target_tensor = torch.cat(rf_targets)

        # calculate similarity probs between val_embs and train_rf_embs
        similarities = []
        for k, val_emb in enumerate(val_emb_tensor):
            similarity_list = []
            for rf_emb_batch in train_rf_embs:
                shape = rf_emb_batch.shape
                val_emb_batch = val_emb.expand(shape[0], shape[1])
                similarity = model.cal_similarity(val_emb_batch, rf_emb_batch)
                similarity_list.append(similarity)
            similarities.append(torch.cat(similarity_list).view(-1))
        sim_matrix = torch.cat(similarities).view(len(val_emb_tensor), -1)
        #sim_matrix, sim_idxes = sim_matrix.sort(dim=1, descending=True)

        # todo:insert new_whale

        # cal map5
        #top5_probs, top5_idxes = sim_matrix.topk(5, dim=1)
        #onehots = (top5_idxes == val_target_tensor.view(-1, 1))
        #onehots = onehots.detach().cpu().numpy()

        #maps = np.zeros(len(onehots))
        #for k, onehot in enumerate(onehots):
        #    r = np.where(onehot == 1)[0]
        #    if r:
        #        maps[k] = 1 / (r[0] + 1)
        map5 = cal_mapk(sim_matrix, val_target_tensor, k=5)
        return map5


class SiameseValidateCallback(fastai.callbacks.tracker.TrackerCallback):
    "A `Callback` to validate SiameseNet."

    def __init__(self, learn):
        super().__init__(learn)
#        class_1_dict = gen_class_reference(self.learn.data.train)
#        self.class_dl = DataLoader(
#            Class1Dataset(self.learn.data.train, class_1_dict),
#            batch_size=self.learn.data.valid_dl.batch_size,
#            shuffle=False,
#            drop_last=False,
#            num_workers=self.learn.data.valid_dl.dl_workers
#        )

    #def on_epoch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    #def on_batch_end(self, last_loss, epoch, num_batch, **kwargs: Any) -> None:
    #    map5 = siamese_validate(self.learn.data.valid_dl, self.learn.model, self.learn.data.train_rf_dl)
    #    print()
    #    return map5

    def on_epoch_end(self, epoch, **kwargs: Any) -> None:
        "Stop the training if necessary."
        map5 = siamese_validate(self.learn.data.valid_dl, self.learn.model, self.learn.data.train_rf_dl)
        print(f'Epoch {epoch}: map5 = {map5}')


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



