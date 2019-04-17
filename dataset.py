from utils import *
import torch
from torch.utils.data.dataset import Dataset
#from albumentations import *
#from albumentations.imgaug import *
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa

def prepare_df(config):
    df = pd.read_csv(config.env.pdir.data/'train.csv')
    new_whale_id = 'z_new_whale'
    change_new_whale(df, new_name=new_whale_id)
    df = filter_df(df, n_new_whale=config.train.new_whale, new_whale_id=new_whale_id)
    df = df.drop('index', 1)
    df_fname = df.set_index('Image')
    val_idxes = split_data_set(df, seed=1)
    #val_idxes = split_whale_idx(df, new_whale_method=(config.train.new_whale!=0), seed=97)
    #val_idxes = split_whale_idx(df, new_whale_method=0, seed=97)
    #val_idxes = split_whale_idx(df, new_whale_method=config.train.new_whale, seed=97)
    all_idxes = list(range(len(df)))
    trn_idxes = list(set(all_idxes) - set(val_idxes))
    labels = sorted(df.Id.unique().tolist())
    if 'new_whale' in labels[-1]:
        labels[-1] = 'new_whale'
        change_new_whale(df, old_name=new_whale_id, new_name='new_whale')
    label2idx = {}
    for k, label in enumerate(labels):
        label2idx[label] = k
    return df, trn_idxes, val_idxes, labels, label2idx

def adjust_edge(bmin, bmax, max_value, adjust_rate=0.05):
    bbox_adjust = int((bmax - bmin) * adjust_rate / 2)
    bmin = max(0, bmin-bbox_adjust)
    bmax = min(max_value, bmax+bbox_adjust)
    return bmin, bmax

def adjust_bbox(bbox, image_shape, adjust_rate=0.05):
    bbox[0], bbox[2] = adjust_edge(bbox[0], bbox[2], image_shape[1], adjust_rate)
    bbox[1], bbox[3] = adjust_edge(bbox[1], bbox[3], image_shape[0], adjust_rate)
    return bbox



class WhaleDataSet(Dataset):
    def __init__(self,
                 config,
                 mode='train',
                 **_):
        self.bboxes = pd.read_csv(config.env.bbox_path)
        self.bboxes = self.bboxes.set_index('Image')
        self.df, self.trn_idxes, self.val_idxes, self.labels, self.label2idx = prepare_df(config)
        self.mode = mode
        #self.c = len(self.df.Id.unique())
        self.c = len(self.labels)
        self.classes = self.labels

        train_seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.AverageBlur(k=(3,3))),
            iaa.Sometimes(0.5, iaa.MotionBlur(k=(3,5))),
            #iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 3.0))),
            #iaa.AddToHueAndSaturation((-20, 20)),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            #iaa.Dropout(p=(0, 0.2)),
            #iaa.CoarseDropout(0.02, size_percent=0.5),
            #iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
            iaa.Sometimes(0.7, iaa.Affine(
                scale={'x': (0.9,1.1), 'y': (0.9,1.1)},
                translate_percent={'x': (-0.05,0.05), 'y': (-0.05,0.05)},
                shear=(-10,10),
                rotate=(-10,10)
                )),
            iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.8,1.0))),
            ], random_order=True)

        if mode == 'train':
            self.idxes = self.trn_idxes
            self.data_path=Path(config.env.pdir.data/'train')
            #self.x = []
            #self.y = []
            #for idx in self.idxes:
            #    self.x.append(idx)
            #    self.y.append(idx)
            self.trn_trfm = train_seq.to_deterministic()
        elif mode == 'val':
            self.idxes = self.val_idxes
            self.data_path=Path(config.env.pdir.data/'train')
        elif mode == 'test':
            self.test_list = sorted(list(Path(config.env.pdir.data/'test').glob('*.jpg')))
            self.idxes = list(range(len(self.test_list)))
            self.data_path=Path(config.env.pdir.data/'test')
        self.len = len(self.idxes)


    def __len__(self):
        return len(self.idxes)

    def get_label(self, index):
        return self.label2idx[self.df.loc[index, 'Id']]

    def __getitem__(self, index):
        index %= self.len
        index = self.idxes[index]
        if self.mode == 'test':
            fname = Path(self.test_list[index])
            bname = fname.name
        else:
            bname = self.df.loc[index, 'Image']
            fname = self.data_path/bname

        #image = cv2.imread(str(fname))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = open_image(fname)
        bbox = self.bboxes.loc[bname].tolist()
        bbox = adjust_bbox(bbox, image.shape[:2], adjust_rate=0.15)
        #bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        #visualize_bbox(image, bbox, (255, 0, 0), 2)
        #plt.imshow(image)


        #show_image(image)
        assert image.ndim == 3
        assert image.shape[-1] == 3

        label_idx = 0
        if self.mode != 'test':
            label_idx = self.get_label(index)

        #transform
        '''
        #trans = Rotate(limit=90, p=1.0).apply(trans)
        #trans = IAAAffine(rotate=40, always_apply=True)(trans)
        #trans = ShiftScaleRotate(shift_limit=0.0, scale_limit=0.5, rotate_limit=90, p=1.0).apply(trans)
        trans = Crop(*bbox).apply(trans)
        #trans = Resize(SZ, SZ).apply(trans)
        '''

        trans = image
        if self.mode == 'train':
            trans = self.trn_trfm.augment_images([image])[0]
        trans = Crop(*bbox).apply(trans)
        trans = Resize(SZ, SZ).apply(trans)
        trans = Normalize().apply(trans)

        #inv = denormalize(trans, imagenet_means, imagenet_std).astype(np.uint8)
        #diff = inv - raw

        #show_image_pil(image)
        #show_image_pil(trans)
        #show_image_pil(trans.transpose(2, 0, 1))
        #show_image(image)
        #show_image(trans)

        #data = self.transform(image=image, bboxes=bbox)
        return [trans.transpose(2, 0, 1), label_idx]


#trn_trfm = Compose([
#    Crop(),
#    Resize(SZ, SZ),
#    #RandomCrop(224, 224),
#    #Normalize(
#    #    mean=[0.485, 0.456, 0.406],
#    #    std=[0.229, 0.224, 0.225],
#    #),
#    #ToTensor()
#])


if __name__ == '__main__':
    image = open_image('/media/wb/backup/work/whale/input/test/0a0ec5a23.jpg')
    show_image(image)

    data = trn_trfm(image=image)
    trfm_image = Image.fromarray(data['image'])
    show_image(data['image'])
    trfm_image.show()
