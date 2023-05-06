import os
import pickle
import random

import numpy as np
import torch

from utils import DATA_INFO_PATH, SAVE_ROOT_PATH, Logger


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric + ': ' + str(value)) + ' '
    return result_str


def flatten_dict(data: dict, join_key: bool = True, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary.

    Args:
        data (dict): data to be flattened.
        join_key (bool, optional): whether join key with parent key and sep. Defaults to True.
        parent_key (str, optional): parent_key. Defaults to ''.
        sep (str, optional): sep. Defaults to '.'.

    Returns:
        dict: flattened dict.
    """
    flattened_dict = {}
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key and join_key else k
        if isinstance(v, dict):
            flattened_dict.update(flatten_dict(v, join_key, new_key, sep))
        else:
            flattened_dict[new_key] = v
    return flattened_dict


def seed_everything(seed: int) -> None:
    """seed everything.

    Args:
        seed (int): seed to be set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created")
    else:
        print(f"Directory {path} already exists")


def filter_tracked_hparams(hparams):
    tracked_hparams = {}
    for parm, val in hparams.items():
        if isinstance(val, (int, float, str, bool, torch.Tensor)):
            tracked_hparams[parm] = val
    return tracked_hparams

def create_logger(dataset, model, commit):
    # XXX too ugly!!!
    logpath = SAVE_ROOT_PATH.format(
        dataset=dataset, model=model, commit=commit)
    create_directory(logpath)
    return Logger(os.path.join(logpath, 'run.log'))


def load_dataset_info(dataset):
    # XXX too ugly!!!
    with open(DATA_INFO_PATH.format(dataset=dataset), 'rb') as f:
        info = pickle.load(f)
        extra_config = {
            'user_num': info['user_num'], 'item_num': info['item_num']}
    return extra_config
