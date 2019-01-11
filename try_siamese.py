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
id2file = make_whale_id_dict(df0)
file2id = df0.set_index('Image').to_dict()

train_item_list = np.asarray(train_list)
val_item_list = np.asarray(val_list)

# data1 = ImageItemList.from_folder('./data/train')
# data2 = data1.split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
# data3 = data2.label_from_func(lambda path: fn2label[path2fn(str(path))])


class DataLoaderEx(DeviceDataLoader):
    def __post_init__(self):
        super().__post_init__()

    def proc_batch(self, b):
        return b

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


class SiameseDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.whale_ids = ds.y.items
        self.id_dict = self.make_id_dict()
        self.len = len(self.ds)

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
        idx_a = idx
        idx_p, type = self.find_same(idx)
        idx_n = self.find_different(idx)
        return self.ds[idx_a][0], self.ds[idx_p][0], self.ds[idx_n][0], type

    def find_same(self, idx):
        whale_id = self.whale_ids[idx]
        c = self.id_dict[whale_id]

        # for 1-class or new_whale, there is no same
        if len(c) == 1 or len(c) > 100:
            return idx, 0   #0 means no anchor-positive pair

        candidates = c[c!=idx]
        np.random.shuffle(candidates)
        return candidates[0], 1

    def find_different(self, idx1, idx2=None):
        #already have idx2
        if idx2 is not None:
            return idx2

        whale_id = self.whale_ids[idx1]
        while True:
            idx2 = np.random.randint(self.len//2)
            if self.whale_ids[idx2] != whale_id:
                break
        return idx2


class TensorDict(object):
    def __init__(self):
        self.dict = {}

    def to(self, device):
        for key in self.dict:
            self.dict[key] = self.dict[key].to(device)

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

    #batch = []
    #batch.append(torch.tensor(np.stack(anchor_list)).to(device))
    #batch.append(torch.tensor(np.stack(pos_list)).to(device))
    #batch.append(torch.tensor(np.stack(neg_list)).to(device))
    #batch.append(torch.tensor(np.stack(pos_valid_masks)).to(device))
    #return batch, batch[-1] # just for (data, target) format

    batch = {}
    batch['anchors'] = torch.tensor(np.stack(anchor_list)).to(device)
    batch['pos_ims'] = torch.tensor(np.stack(pos_list)).to(device)
    batch['neg_ims'] = torch.tensor(np.stack(neg_list)).to(device)
    batch['pos_valid_masks'] = torch.tensor(np.stack(pos_valid_masks)).to(device)
    batch_len = len(batch['anchors'])
    target = torch.cat((torch.ones(batch_len, dtype=torch.int32), torch.zeros(batch_len, dtype=torch.int32)))
    return batch, target


train_dl = DataLoader(
    SiameseDataset(data.train),
    batch_size=BS,
    shuffle=True,
    #collate_fn=siamese_collate,
    num_workers=NUM_WORKERS
)

valid_dl = DataLoader(
    SiameseDataset(data.valid),
    batch_size=BS,
    shuffle=False,
    #collate_fn=siamese_collate,
    num_workers=NUM_WORKERS
)

data_bunch = ImageDataBunch(train_dl, valid_dl, collate_fn=siamese_collate)
#data_bunch = DataBunch(train_dl, valid_dl, collate_fn=siamese_collate)
data_bunch.train_dl = DataLoaderEx(train_dl, None, None, siamese_collate)
data_bunch.valid_dl = DataLoaderEx(valid_dl, None, None, siamese_collate)

def normalize_batch(batch):
    stat_tensors = [torch.tensor(l).to(device) for l in imagenet_stats]
    return [normalize(batch[0][0], *stat_tensors), normalize(batch[0][1], *stat_tensors)], batch[1]

data_bunch.add_tfm(normalize_batch)


#for train_batch, target in data_bunch.train_dl:
#    print(len(train_batch), len(train_batch[0]))
#    break
#exit()


def cnn_activations_count(model):
    _, ch, h, w = model_sizes(create_body(models.resnet18), (SZ, SZ))[-1]
    return ch * h * w


class Siamese(nn.Module):
    def __init__(self, lin_ftrs=128, arch=models.resnet18, norm=1):
        super().__init__()
        self.cnn = create_body(arch)
        self.fc1 = nn.Linear(cnn_activations_count(self.cnn), lin_ftrs)
        self.fc2 = nn.Linear(lin_ftrs, 1)
        self.norm = norm

    def siamese_emb(self, im):
        x = self.cnn(im)
        x = self.process_features(x)
        x = self.fc1(x)
        emb = torch.sigmoid(x)
        return emb

    def distance(self, a, b, norm=1):
        if self.norm == 1:
            return torch.abs(a - b)
        else:
            return (a - b) ** 2

    def forward_test(self, im_a, im_b):
        emb_a = self.siamese_emb(im_a)
        emb_b = self.siamese_emb(im_b)
        dist_v = self.distance(emb_a, emb_b)
        prob = torch.sigmoid(self.fc2(dist_v))
        return prob

    def forward_train(self, batch):
        anchor_emb = self.siamese_emb(batch['anchors'])
        pos_emb = self.siamese_emb(batch['pos_ims'])
        neg_emb = self.siamese_emb(batch['neg_ims'])

        pos_dist = self.distance(anchor_emb, pos_emb)
        neg_dist = self.distance(anchor_emb, neg_emb)

        pos_logits = self.fc2(pos_dist)
        neg_logits = self.fc2(neg_dist)
        #return pos_logits, neg_logits, pos_dist, neg_dist
        return torch.cat((pos_logits, neg_logits))

    def forward(self, batch):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_test(batch)

    def process_features(self, x): return x.reshape(x.shape[0], -1)


siamese = Siamese()

# print(type(data.__getitem__(3)))

learn = Learner(data_bunch, siamese, loss_func=BCEWithLogitsFlat(), metrics=[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)])

learn.fit_one_cycle(2)

pass
