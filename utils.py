import numpy as np
import torch
import pandas as pd

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def split_whale_set(df, nth_fold=0, total_folds=5, new_whale_method=0, seed=1):
    '''
    Split whale dataset to train and valid set based on k-fold idea.
    total_folds: number of total folds
    nth_fold: the nth fold
    new_whale_method: If 0, remove new_whale in all data sets; if 1, add new_whale to train/validation sets
    seed: Random seed for shuffling
    '''
    np.random.seed(seed)
    #list(df_known.groupby('Id'))
    train_list = []
    val_list = []
    #df_known = df[df.Id!='new_whale']
    for name, group in df.groupby('Id'):
        #print(name, len(group), group.index, type(group))
        #if name == 'w_b82d0eb':
        #    print(name, df_known[df_known.Id==name])
        if new_whale_method == 0 and name == 'new_whale':
            continue
        group_num = len(group)
        images = group.Image.values
        if group_num > 1:
            np.random.shuffle(images)
            #images = list(images)
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
    #list(df_known.groupby('Id'))
    train_list = []
    val_list = []
    df_known = df[df.Id!='new_whale']
    for name, group in df_known.groupby('Id'):
        #print(name, len(group), group.index, type(group))
        #if name == 'w_b82d0eb':
        #    print(name, df_known[df_known.Id==name])
        group_num = len(group)
        idxes = group.index.values
        if group_num > 1:
            np.random.shuffle(idxes)
            #idxes = list(idxes)
            span = max(1, group_num // total_split)
            val_idxes = idxes[nth_split*span:(nth_split+1)*span]
            train_idxes = list(set(idxes) - set(val_idxes))
            val_list.extend(val_idxes)
            train_list.extend(train_idxes)
        else:
            train_list.extend(idxes)

    if use_new_whale:
        df_new = df[df.Id=='new_whale']
        train_list.extend(df_new.index.values)

    return train_list, val_list


def make_whale_id_dict(df):
    whale_id_dict = {}
    for name, group in df.groupby('Id'):
        whale_id_dict[name] = group.Image.tolist()
    return whale_id_dict



# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

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



