import sys
sys.path.append("..")

from PIL import Image
from common import *
from utils import *
from dataset import *

def test_trfm():

    image = open_image('../../input/test/0a0ec5a23.jpg')
    show_image(image)

    data = trn_trfm(image=image)
    trfm_image = Image.fromarray(data['image'])
    show_image(trfm_image)


