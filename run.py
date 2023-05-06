import argparse
import os

import models
from config import Config
from trainer import BaseLearner, SequentialDataLoader, learner_dict
from utils import (create_logger, load_dataset_info,
                   seed_everything)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="Beauty")
parser.add_argument('--model', type=str, default="MARIS")
parser.add_argument('--sample_strategy', default="sample_sort_bpr",
                    choices=["full_sort_ce", "full_sort_bpr", "sample_sort_ce", "sample_sort_bpr"])
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--commit', type=str, default='base')

args = parser.parse_args()

if __name__ == "__main__":

    # Create a logger object to log the training process
    logger = create_logger(args.dataset, args.model, args.commit)

    # Load additional configuration information for the specified dataset
    extra_config = load_dataset_info(args.dataset)
    args = Config(args, extra_config=extra_config)
    config = args.easy_use()

    # Set the seed for random number generation
    seed_everything(2048)

    # Create the loader, model and learner object
    loader = SequentialDataLoader(config, logger)
    model_cls = getattr(models, config.model)
    learner_cls = learner_dict.get(config.model, BaseLearner)

    # Instantiate the model object and move it to the specified device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    model = model_cls(config, logger)
    model.cuda()

    # Instantiate the learner object and pass in the configuration, model, data loader, and logger
    learner = learner_cls(config, model, loader, logger)

    # Print the configuration to the logger
    args.tab_printer(logger)

    # Train the model
    learner.train()

    # Test the model
    learner.test()
