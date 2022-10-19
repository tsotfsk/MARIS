import json
import os
import torch
from texttable import Texttable


class EasyDict():
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def to_parm_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, torch.Tensor)):
                result[key] = value
            elif isinstance(value, dict):
                for lkey, lvalue in value.items():
                    if isinstance(lvalue, (str, int, float, bool, torch.Tensor)):
                        result[lkey] = lvalue
        return result


class Config(object):

    def __init__(self, args):
        self.confpath = os.path.join(args.confpath, args.dataset,
                                     f'{args.model.lower()}.json')

        self.config_in_files = self.load_config()
        self.cofnig_in_cmd = vars(args)

        self.config = self.merge_config()

    def easy_use(self):
        return EasyDict(self.config)

    def load_config(self):
        return json.load(open(self.confpath, 'r'))

    def merge_config(self):
        config = {}
        config.update(self.config_in_files)
        config.update(self.cofnig_in_cmd)
        if (config['method']) and ("method" in config['model_param']):
            config['model_param']['method'] = config['method']
            del config['method']
        return config

    def tab_printer(self, logger):
        """
        Function to print the logs in a nice tabular format.
        :param args: Parameters used for the model.
        """
        args = self.config
        keys = sorted(args.keys())
        t = Texttable()
        t.add_rows([["Parameter", "Value"]] +
                   [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
        logger.info('\n' + t.draw())
