
from utils import *
from fastai.callbacks import *

categories = 20
num_per_cat = 3

src_train_dir = '../input/train/'
dst_train_dir = '../input/train1/'
src_test_dir = '../input/test/'
dst_test_dir = '../input/test1/'

os.makedirs(dst_train_dir, exist_ok=True)
os.makedirs(dst_test_dir, exist_ok=True)

df = pd.read_csv('../input/train.csv')

train_list = []
val_list = []
for k, (name, group) in enumerate(df.groupby('Id')):
    # print(name, len(group), group.index, type(group))
    # if name == 'w_b82d0eb':
    #    print(name, df_known[df_known.Id==name])
    group_num = len(group)
    images = group.Image.values
    if group_num > 1:
        num = min(group_num//2, num_per_cat)
        train_list.extend(images[:num])
        val_list.extend(images[-num:])
    else:
        train_list.append(images[0])

    if k == categories:
        break

for item in val_list:
    if item in train_list:
        print('train_list', len(train_list), train_list)
        print('val_list', len(val_list), val_list)
        print(f'Error: {item} in train_list')
        exit()

for f in train_list + val_list:
    f = shutil.copyfile(src_train_dir+f, dst_train_dir+f)

print('train_list', len(train_list), train_list)
print('val_list', len(val_list), val_list)



