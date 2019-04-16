import pickle
import datetime as dt
from easydict import EasyDict
import yaml
import logging
import pathlib
from pathlib import Path
import numpy as np
import PIL
import matplotlib.pyplot as plt
import cv2

def load_config(config_file):
    with open(config_file, 'r') as fid:
        yaml_config = EasyDict(yaml.load(fid))
    return yaml_config

def now2str(format="%Y-%m-%d__%H-%M-%S"):
    # str_time = time.strftime("%Y-%b-%d-%H-%M-%S", time.localtime(time.time()))
    return dt.datetime.now().strftime(format)

def save_dump(dump_data, out_file):
    with open(out_file, 'wb') as fp:
        print('Writing to %s.' % out_file)
        #pickle.dump(dump_data, fp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dump_data, fp)

def load_dump(dump_file):
    with open(dump_file, 'rb') as fp:
        dump = pickle.load(fp)
        return dump

def init_logger(name='qf', to_console=True, log_file=None, level=logging.DEBUG,
                msg_fmt='[%(asctime)s]  %(message)s', time_fmt="%Y-%m-%d %H:%M:%S"):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(msg_fmt, time_fmt)

    if logger.handlers != [] and isinstance(logger.handlers[0], logging.StreamHandler):
        logger.handlers.pop(0)
    # create console handler and set level to debug
    f = open("/tmp/debug", "w")          # example handler
    if to_console:
        f = None

    ch = logging.StreamHandler(f)
    ch.setLevel(level)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode='a', encoding=None, delay=False)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def linear_decay(step, pars):
    start_value = pars[0]
    end_value = pars[1]
    start_step = pars[2]
    end_step = pars[3]
    assert start_step <= end_step

    if step < start_step:
        return 0.0
    elif step >= end_step:
        return end_value
    #epsilon = max(end_value, start_value - (step - start_step) * (start_value - end_value) / (end_step - start_step))
    return start_value - (step - start_step) * (start_value - end_value) / (end_step - start_step)


class BaseDirs():
    def __init__(self, root_path='/media/wb/backup/work/whale', data_path='input'):
        dir_list = []
        if root_path == '':
            self.root = Path().resolve().parent
        else:
            self.root = Path(root_path)

        self.data = self.root/data_path
        self.data.mkdir(exist_ok=True)

        self.models = self.root/'models'
        self.models.mkdir(exist_ok=True)

        self.runtime = self.root/'runtime'
        self.runtime.mkdir(exist_ok=True)

        self.log = self.root/'log'
        self.log.mkdir(exist_ok=True)

        self.tb_log = self.root/'tblog'
        self.tb_log.mkdir(exist_ok=True)

        self.tmp = self.root/'tmp'
        self.tmp.mkdir(exist_ok=True)

    def add_dir(self, base_dir, sub_dir):
        new_dir = setattr(base_dir, sub_dir, f'{base_dir}/{sub_dir}')
        new_dir.mkdir(exist_ok=True)

start_timestamp = now2str()

pdir = BaseDirs()
plog_file = pdir.log/f'{start_timestamp}.log'
plog = init_logger(log_file=plog_file)

#todo scoreboard
class Scoreboard():
    def __init__(self, sb_file, sb_len=11, sort='dec'):
        if sb_file.is_file():
            load_obj = load_dump(sb_file)
            self.__dict__.update(load_obj.__dict__)
        else:
            self.sb = []
            self.sb_len = sb_len
            self.sort = sort
            self.sb_file = sb_file

    def update(self, content:dict):
        self.sb.append(content)
        reverse = self.sort in 'decrease'
        self.sb.sort(key=lambda e: e['score'], reverse=reverse)

        #remove useless files
        if len(self.sb) > self.sb_len:
            del_file = self.sb[-1]['file']
            if del_file.is_file():
                del_file.unlink()
        self.sb = self.sb[:self.sb_len]

        save_dump(self, self.sb_file)

    def __len__(self):
        return len(self.sb)

    def __getitem__(self, idx):
        return self.sb[idx]

    def is_full(self):
        return len(self.sb) >= self.sb_len



imagenet_means, imagenet_std = map(np.array, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
#ds_means, ds_std = map(np.array, ([0.467, 0.467, 0.467], [0.163, 0.163, 0.163]))
ds_means = imagenet_means
ds_std = imagenet_std

# normalize = lambda x: (x - imagenet_means) / imagenet_std
# denormalize = lambda x: x * imagenet_std + imagenet_means
#normalize = lambda x: (x - ds_means * 255) / (ds_std * 255)
#denormalize = lambda x: int(x * ds_std * 255 + ds_means * 255)

def denormalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    img = img.astype(np.float32)
    img *= std
    img += mean
    return img.astype(np.uint8)

def open_image(fn):
    x = PIL.Image.open(fn).convert('RGB')
    return np.asarray(x)

def show_image_pil(im):
    if im.shape[0] == 3:
        im = im.transpose(1,2,0)
    if im.min() < 0 and im.ndim == 3:
        im=denormalize(im, imagenet_means, imagenet_std)
    img = PIL.Image.fromarray(im, 'RGB')
    img.show()

def show_image(im, figsize=None, ax=None, alpha=None):
    if im.shape[0] == 3: im = im.transpose(1,2,0)
    if im.min() < 0 and im.ndim == 3: im=denormalize(im); im = np.clip(im, 0, 1) # this is quite horrible and can lead to bugs
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=thickness)
    return img

def visualize_bbox1(img, bbox, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)

