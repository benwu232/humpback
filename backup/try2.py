from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
from fastai.callbacks.hooks import num_features_model, model_sizes

from utils import *
import fastai

SZ = 224
BS = 16
NUM_WORKERS = 1
SEED = 1

root_path = '../input/'
train_path = '../input/train/'
test_path = '../input/test/'

df0 = pd.read_csv('../input/train.csv')
df_new = df0[df0.Id == 'new_whale']
df_known = df0[df0.Id != 'new_whale']
train_list, val_list = split_whale_set(df0, nth_fold=0, new_whale_method=1, seed=1)
print(len(train_list), len(val_list))

im_count = df0[df0.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
ex_df = df0.join(im_count, on='Id')

path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
fn2label = {row[1].Image: row[1].Id for row in df0.iterrows()}
id2file = make_whale_class_dict(df0)
file2id = df0.set_index('Image').to_dict()

train_item_list = np.asarray(train_list)
val_item_list = np.asarray(val_list)

# data1 = ImageItemList.from_folder('./data/train')
# data2 = data1.split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
# data3 = data2.label_from_func(lambda path: fn2label[path2fn(str(path))])

data = (
    ImageItemList
        # .from_df(df_known, 'data/train', cols=['Image'])
        .from_folder(train_path)
        # .split_by_idxs(train_item_list, val_item_list)
        .split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
        # .split_by_idx(val_list)
        # .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path2fn(str(path))])
        .add_test(ImageItemList.from_folder(test_path))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ,
                   resize_method=ResizeMethod.SQUISH)
        #.databunch(bs=BS, num_workers=NUM_WORKERS, path=root_path)
        #.normalize(imagenet_stats)
)


def is_even(num): return num % 2 == 0


class TwoImDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.whale_ids = ds.y.items
        self.id_dict = self.make_id_dict()
        self.len = 2 * len(self.ds)

    def make_id_dict(self):
        id_dict = {}
        for k, id in enumerate(self.whale_ids):
            if id not in id_dict:
                id_dict[id] = [k]
            else:
                id_dict[id].append(k)

        for id in id_dict:
            id_dict[id] = np.asarray(id_dict[id])

        return id_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.sample_same(idx // 2)
        else:
            return self.sample_different((idx - 1) // 2)

    def sample_same(self, idx):
        whale_id = self.whale_ids[idx]
        c = self.id_dict[whale_id]

        # for 1-class or new_whale, there is no same
        if len(c) == 1 or len(c) > 100:
            result = self.sample_different(idx)
            result[-1] = 2
            return result

        candidates = c[c!=idx]
        np.random.shuffle(candidates)
        return [self.ds[idx][0], self.ds[candidates[0]][0], 1]

    def sample_different(self, idx1, idx2=None):
        #already have idx2
        if idx2 is not None:
            return [self.ds[idx1][0], self.ds[idx2][0], 0]

        whale_id = self.whale_ids[idx1]
        while True:
            idx2 = np.random.randint(self.len//2)
            if self.whale_ids[idx2] != whale_id:
                break
        return [self.ds[idx1][0], self.ds[idx2][0], 0]

    def sample_different1(self, idx):
        whale_id = self.whale_ids[idx]
        candidates = list(np.where(self.whale_ids != whale_id)[0])
        np.random.shuffle(candidates)
        return self.construct_example(self.ds[idx][0], self.ds[candidates[0]][0], 0)

    def construct_example(self, im_A, im_B, class_idx):
        return [im_A, im_B], class_idx


train_dl = DataLoader(
    TwoImDataset(data.train),
    batch_size=BS,
    shuffle=True,
    num_workers=NUM_WORKERS
)
valid_dl = DataLoader(
    TwoImDataset(data.valid),
    batch_size=BS,
    shuffle=False,
    num_workers=NUM_WORKERS
)

data_bunch = ImageDataBunch(train_dl, valid_dl)

def normalize_batch(batch):
    stat_tensors = [torch.tensor(l).to(device) for l in imagenet_stats]
    return [normalize(batch[0][0], *stat_tensors), normalize(batch[0][1], *stat_tensors)], batch[1]

data_bunch.add_tfm(normalize_batch)

#for train_batch in train_dl:
#    print(train_batch)
#    break


def cnn_activations_count(model):
    _, ch, h, w = model_sizes(create_body(models.resnet18), (SZ, SZ))[-1]
    return ch * h * w


class Siamese(nn.Module):
    def __init__(self, lin_ftrs=128, arch=models.resnet18):
        super().__init__()
        self.cnn = create_body(arch)
        self.fc1 = nn.Linear(cnn_activations_count(self.cnn), lin_ftrs)
        self.fc2 = nn.Linear(lin_ftrs, 1)

    def forward(self, im_a, im_b):
        xa = self.cnn(im_a)
        xa = self.process_features(xa)
        xa = self.fc1(xa)
        emb_a = torch.sigmoid(xa)

        xb = self.cnn(im_b)
        xb = self.process_features(xb)
        xb = self.fc1(xb)
        emb_b = torch.sigmoid(xb)

        distance = (emb_a - emb_b) ** 2
        prob = torch.sigmoid(self.fc2(distance))
        return prob


    def calculate_distance(self, x1, x2): return (x1 - x2).abs_()
    def process_features(self, x): return x.reshape(x.shape[0], -1)


siamese = Siamese()

# print(type(data.__getitem__(3)))

learn = Learner(data_bunch, siamese, loss_func=BCEWithLogitsFlat(), metrics=[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)])

learn.fit_one_cycle(2)

pass
