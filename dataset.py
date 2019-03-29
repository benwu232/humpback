
from common import *
from utils import *
import torch
from torch.utils.data.dataset import Dataset

def prepare_df(config):
    df = pd.read_csv(LABELS)
    new_whale_id = 'z_new_whale'
    change_new_whale(df, new_whale_id)
    df = filter_df(df, n_new_whale=config.train.new_whale, new_whale_id=new_whale_id)
    df = df.drop('index', 1)
    df_fname = df.set_index('Image')
    #val_idxes = split_data_set(df, seed=1)
    #val_idxes = split_whale_idx(df, new_whale_method=(config.train.new_whale!=0), seed=97)
    #val_idxes = split_whale_idx(df, new_whale_method=0, seed=97)
    val_idxes = split_whale_idx(df, new_whale_method=config.train.new_whale, seed=97)
    all_idxes = list(range(len(df)))
    trn_idxes = list(set(all_idxes) - set(val_idxes))
    labels = sorted(df.Id.unique().tolist())
    labels[-1] = 'new_whale'
    label2idx = {}
    for k, label in enumerate(labels):
        label2idx[label] = k
    return df, trn_idxes, val_idxes, labels, label2idx

class WhaleDataSet(Dataset):
    def __init__(self,
                 config,
                 df,
                 idxes,
                 labels,
                 label2idx,
                 data_path=TRAIN,
                 #dataset_dir,
                 #split,
                 transform=None,
                 #landmark_ver='5',
                 train_csv='train.csv',
                 **_):
        #self.split = split
        #self.transform = transform
        #self.dataset_dir = dataset_dir
        #self.landmark_ver = landmark_ver
        #self.train_csv = 'train.csv'
        self.df = df
        self.idxes = idxes
        self.labels = labels
        self.label2idx = label2idx
        self.data_path = Path(data_path)
        self.len = len(self.idxes)
        self.transform = None

    def __len__(self):
        return len(self.idxes)

    def get_label(self, index):
        return self.label2idx[self.df.loc[index, 'Id']]

    def __getitem__(self, index):
        index %= self.len
        fname = self.df.loc[index, 'Image']
        fname = self.data_path/fname

        image = open_image(fname)
        #show_image(image)
        assert image.ndim == 3
        assert image.shape[-1] == 3

        label_idx = self.get_label(index)
        if self.transform is not None:
            data = self.transform(image, box, landmark, example['Image'])

        return [data, label_idx]


