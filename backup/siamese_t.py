
from utils import *
from fastai.callbacks import *
import torchvision

torch.multiprocessing.set_sharing_strategy('file_system')

#some parameters
debug = 0
enable_lr_find = 1
now_str = now2str(format="%Y-%m-%d_%H-%M-%S")
txlog = tx.SummaryWriter(f'../tblog/{now_str}')

arch = models.resnet18
im_size = 224
if debug == 1:
    #arch = torchvision.models.squeezenet1_1(pretrained=True)
    train_batch_size = 8
    val_batch_size = 8
    dl_workers = 0
else:
    train_batch_size = 64
    val_batch_size = 256
    dl_workers = 4
seed = 1

emb_len = 128
diff_method = 1

root_path = '../input/'
if debug == 1:
    train_path = '../input/train1_224/'
    test_path = '../input/test1_224/'
elif debug == 2:
    train_path = '../input/train2_224/'
    test_path = '../input/test2_224/'
else:
    train_path = '../input/train_224/'
    test_path = '../input/test_224/'


df0 = pd.read_csv('../input/train.csv')
df_new = df0[df0.Id == 'new_whale']
df_known = df0[df0.Id != 'new_whale']
if debug:
    train_list = train_list_dbg
    val_list = val_list_dbg
else:
    train_list, val_list = split_whale_set(df0, nth_fold=0, new_whale_method=1, seed=1)
print(len(train_list), len(val_list))

im_count = df0[df0.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
ex_df = df0.join(im_count, on='Id')

path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)
fn2label = {row[1].Image: row[1].Id for row in df0.iterrows()}
class_dict = make_whale_class_dict(df0)
file_lut = df0.set_index('Image').to_dict()

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
        .transform([None, None], size=im_size, resize_method=ResizeMethod.SQUISH)
        #.transform(im_tfms, size=im_size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=train_batch_size, num_workers=dl_workers, path=root_path)
        #.normalize(imagenet_stats)
)

data.add_tfm(normalize_batch)

train_dl = DataLoader(
    SiameseDs(data.train_ds),
    batch_size=train_batch_size,
    shuffle=True,
    #collate_fn=siamese_collate,
    num_workers=dl_workers
)

data_v = (
    ImageItemList
        # .from_df(df_known, 'data/train', cols=['Image'])
        .from_folder(train_path)
        # .split_by_idxs(train_item_list, val_item_list)
        .split_by_valid_func(lambda path: path2fn(str(path)) in val_list)
        # .split_by_idx(val_list)
        # .random_split_by_pct(seed=SEED)
        .label_from_func(lambda path: fn2label[path2fn(str(path))])
        .add_test(ImageItemList.from_folder(test_path))
        #.transform([None, None], size=im_size, resize_method=ResizeMethod.SQUISH)
        #.transform(im_tfms, size=im_size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=train_batch_size, num_workers=dl_workers, path=root_path)
        .normalize(imagenet_stats)
)

#v = SimpleDataset(data.valid)
valid_dl = DataLoader(
    SimpleDataset(data_v.valid_ds),
    batch_size=val_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=dl_workers
)

test_dl = DataLoader(
    SimpleDataset(data_v.test_ds),
    batch_size=val_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=dl_workers
)

ref_dl = DataLoader(
    SimpleDataset(data_v.train_ds),
    batch_size=val_batch_size,
    shuffle=False,
    #collate_fn=siamese_collate,
    num_workers=dl_workers
)

data_bunch = ImageDataBunch(train_dl, valid_dl, fix_dl=ref_dl)

'''
data_bunch.train_dl = DataLoaderTrain1(train_dl, device, tfms=im_tfms[0], collate_fn=collate_siamese)
#data_bunch.valid_dl = DataLoaderMod(valid_dl, None, None, siamese_collate)
data_bunch.valid_dl = DataLoaderVal(valid_dl, device, tfms=None, collate_fn=data_collate)
data_bunch.test_dl = DataLoaderVal(test_dl, device, tfms=None, collate_fn=data_collate)
data_bunch.fix_dl = DataLoaderVal(ref_dl, device, tfms=None, collate_fn=data_collate)
#data_bunch.add_tfm(normalize_batch)
#data_bunch.valid_dl = None
'''

#for batch in data_bunch.train_dl:
#    print(len(batch))
#    break
#for batch in data_bunch.fix_dl:
#    print(len(batch))
#    break
#
#exit()

#siamese = SiameseNet(emb_len, arch=arch, width=im_size, height=im_size, diff_method=diff_method)
siamese = SiameseNetwork(arch=arch)

# new_whale should not be involved in positive distance
new_whale_idx = find_new_whale_idx(data.train_ds.y.classes)
#new_whale_idx = find_new_whale_idx(data_v.valid_ds.y.classes)
#triploss = TripletLoss(margin=0.2)
contrastive_loss = ContrastiveLoss(margin=1.0)

learn = LearnerEx(data_bunch,
                  siamese,
                  enable_validate=False,
                  loss_func=BCEWithLogitsFlat(),
                  #loss_func=contrastive_loss,
                  #metrics=[lambda preds, targs: accuracy_thresh(preds.squeeze(), targs, sigmoid=False)]
                  )

learn.load(f'res18-siamese-stage-2')
learn.model.to(device)

map5, top5_matrix, pos_dist_max, neg_dist_min = siamese_mat(valid_dl, learn.model, ref_dl,
                                                            pos_mask=[0], ref_idx2class=ref_dl.ds.y,
                                                            target_idx2class=valid_dl.ds.y)

print(map5, pos_dist_max, neg_dist_min)

