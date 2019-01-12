import torch

from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
from fastai.callbacks.hooks import num_features_model, model_sizes

from utils import *
import fastai

#torch.multiprocessing.set_start_method('forkserver', force=True)

SZ = 224
BS = 64
NUM_WORKERS = 6
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
data_bunch.train_dl = DataLoaderMod(train_dl, None, None, siamese_collate)
data_bunch.valid_dl = DataLoaderMod(valid_dl, None, None, siamese_collate)

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
