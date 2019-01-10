from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd

from utils import *
import fastai

def split_whale_set(df, nth_fold=0, total_folds=5, new_whale_method=0, seed=1):
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

SZ = 224
BS = 16
NUM_WORKERS = 1
SEED=1

root_path = '../input/'
train_path = '../input/train/'
test_path = '../input/test/'

ori_df = pd.read_csv('../input/train.csv')
df_new = ori_df[ori_df.Id=='new_whale']
df_known = ori_df[ori_df.Id!='new_whale']
train_list, val_list = split_whale_set(ori_df, nth_fold=0, new_whale_method=0, seed=1)
print(len(train_list), len(val_list))

im_count = ori_df[ori_df.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
ex_df = ori_df.join(im_count, on='Id')

path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
fn2label = {row[1].Image: row[1].Id for row in ori_df.iterrows()}

train_item_list = np.asarray(train_list)
val_item_list = np.asarray(val_list)

#data1 = ImageItemList.from_folder('./data/train')
#data2 = data1.split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
#data3 = data2.label_from_func(lambda path: fn2label[path2fn(str(path))])

data = (
    ImageItemList
        #.from_df(df_known, 'data/train', cols=['Image'])
        .from_folder(train_path)
        #.split_by_idxs(train_item_list, val_item_list)
        .split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
        #.split_by_idx(val_list)
        #.random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path2fn(str(path))])
        .add_test(ImageItemList.from_folder(test_path))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path=root_path)
        .normalize(imagenet_stats)
)
#print(type(data.__getitem__(3)))
learn = create_cnn(data, models.resnet50, pretrained=False, metrics=[accuracy, map5])

learn.fit_one_cycle(2)

pass