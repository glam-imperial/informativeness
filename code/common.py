import os

try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import namedtuple


def dict_to_struct(obj):
    obj = namedtuple("Configuration", obj.keys())(*obj.values())
    return obj


def make_dirs_safe(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
