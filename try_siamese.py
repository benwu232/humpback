
from utils import *
from fastai.callbacks import *

#some parameters
arch = models.resnet18
im_size = 224
batch_size = 64
dl_workers = 6
seed = 1

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
class_dict = make_whale_class_dict(df0)
file_lut = df0.set_index('Image').to_dict()

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
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=im_size,
                   resize_method=ResizeMethod.SQUISH)
        #.databunch(bs=BS, num_workers=NUM_WORKERS, path=root_path)
        #.normalize(imagenet_stats)
)

train_dl = DataLoader(
    SiameseDataset(data.train),
    batch_size=batch_size,
    shuffle=True,
    #collate_fn=siamese_collate,
    num_workers=dl_workers
)

#valid_dl = DataLoader(
#    SiameseDataset(data.valid),
#    batch_size=batch_size,
#    shuffle=False,
#    drop_last=False,
#    #collate_fn=siamese_collate,
#    num_workers=dl_workers
#)

#v = SimpleDataset(data.valid)
valid_dl = DataLoader(
    SimpleDataset(data.valid),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=dl_workers
)

class_1_dict = gen_class_reference(data.train)
class_dl = DataLoader(
    Class1Dataset(data.train, class_1_dict),
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=data_collate,
    num_workers=dl_workers
)

class_dl = DataLoaderVal(class_dl, device, collate_fn=data_collate, tfms=)

.transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=im_size,
           resize_method=ResizeMethod.SQUISH)
for batch in class_dl:
    print(len(batch), (batch[0].shape))
    break
exit()


data_bunch = ImageDataBunch(train_dl, valid_dl, collate_fn=siamese_collate)
data_bunch.train_dl = DataLoaderTrain(train_dl, device, None, siamese_collate)
#data_bunch.valid_dl = DataLoaderMod(valid_dl, None, None, siamese_collate)
data_bunch.valid_dl = DataLoaderVal(valid_dl, device, collate_fn=data_collate)
#data_bunch.valid_dl = DeviceDataLoader(valid_dl, device, collate_fn=torch.utils.data.dataloader.default_collate)
data_bunch.add_tfm(normalize_batch)
#data_bunch.valid_dl = None

#for train_batch, target in data_bunch.train_dl:
#    print(len(train_batch), len(train_batch[0]))
#    break
#exit()

siamese = SiameseNet(emb_len=128, arch=arch, width=im_size, height=im_size, norm=1)

learn = LearnerEx(data_bunch,
                siamese,
                enable_validate=False,
                loss_func=BCEWithLogitsFlat(),
                metrics=[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)]
                )

cb_save_model = SaveModelCallback(learn, every="epoch", name=f"siamese")
cb_siamese_validate = SiameseValidateCallback(learn, class_dl)
cbs = [cb_save_model, cb_siamese_validate]

learn.fit_one_cycle(2, callbacks=cbs)

