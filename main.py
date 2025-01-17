"""Main function"""

import numpy as np
import utils
import json
import argparse
import datetime
import time
import config
from experiment import experiment
from relation import populate

parser = argparse.ArgumentParser()
parser.add_argument('--run_experiments', default=False, action='store_true')
parser.add_argument('--run_analysis', default=False, action='store_true')
parser.add_argument('--cpu', default=1, type=int)
parser.add_argument('--error_type', default=None)
parser.add_argument('--seeds', default=None, type=int, nargs='+')
parser.add_argument('--log', default=False, action='store_true')
parser.add_argument('--dataset', default=None)
parser.add_argument('--nosave', default=False, action='store_true')
parser.add_argument('--alpha', default=0.05, type=float)

args = parser.parse_args()

if __name__ == "__main__":
    # run experiments on datasets
    if args.run_experiments:
        datasets = [utils.get_dataset(args.dataset)] if args.dataset is not None else config.datasets
        experiment(datasets, args.log, args.cpu, args.nosave, args.error_type, args.seeds)

    # run analysis on results
    if args.run_analysis:
        populate([args.alpha])
