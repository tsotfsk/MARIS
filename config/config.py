import json
import os
from texttable import Texttable
from box import Box
from utils import flatten_dict, CONFIG_PATH

class Config(object):

    def __init__(self, args, extra_config=None):
        self.confpath = CONFIG_PATH.format(dataset=args.dataset,
                                     model=args.model.lower())

        config_in_files = json.load(open(self.confpath, 'r'))
        config_in_cmd = vars(args)

        # priority: extra < file < cmd
        self.config = {}
        self.config_sources = [extra_config] + [config_in_files, config_in_cmd]
        for conf in self.config_sources:
            self.update_config(conf)

    def easy_use(self):
        return Box(self.config)

    def update_config(self, conf: dict):
        self.config.update(conf)

    def to_parm_dict(self):
        return flatten_dict(self.config, join_key=True, sep="->")

    def tab_printer(self, logger):
        """
        Function to print the logs in a nice tabular format.
        :param args: Parameters used for the model.
        """
        flat_dict = self.to_parm_dict()
        keys = sorted(flat_dict.keys())
        t = Texttable()
        t.add_rows([["Parameter", "Value"]] +
                   [[k.replace("_", " ").capitalize(), flat_dict[k]] for k in keys])
        logger.info('\n' + t.draw())
