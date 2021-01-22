import os

from functools import wraps
from PIL import Image
from time import time


def get_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timeit(func):
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        print(f'func:{func.__name__} took {(te - ts):2.6f} sec')
        return result

    return wrap


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_imagenet_label_dict():
    with open(os.path.join(get_root(), "imagenet1000_clsidx_to_labels.txt")) as f:
        idx2label = eval(f.read())
    return idx2label
