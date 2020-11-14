#!/usr/bin/env python3

"""
Perform model selection.

To use this you do not need any special dependency in your program. All you need is to accept a command line
argument and to output some data on stdout.

Input: hyperparameter grid  (yml configuration file OR hyperopt specification exported from python file)
Output: best hyperparametrization

Application should accept hyperparameters from a --hp command line parameter. The format of the --hp parameter is
a string that contains the JSON/YAML dict with the hyperparameters. Castings may be necessary.
The application must output on stdout this yaml pattern:

    ---
    Validation loss: xxxxx
    Best epoch: xxx
    Additional metrics:
        - training_perf: xxxxx
        - name_without_spaces: floating_point_value_or_single_line_string
        - name_without_spaces: floating_point_value_or_single_line_string
        - ...
    ---

"Validation loss" is the only mandatory line.
Note that the validation loss must decrease by convention (i.e. lower is better).
If your loss L is an accuracy, you can just print -L or 1/L.

"""

import argparse
import json
import math
import os
import ray
import re
import subprocess
import ruamel.yaml as yaml
import sys

import numpy as np
from hyperopt import hp
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch


def load_grid(filename: str):
    """
    Loads a grid and converts it into a hyperopt dict
    :param filename:
    :return:
    """
    grid = None
    if filename.endswith('.yaml') or filename.endswith('.yml'):
        with open(filename, 'r') as f:
            grid = yaml.safe_load(f)  # FIXME Handle YAMLError?
        grid = interpret_grid(grid)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            grid = json.load(f)  # FIXME Handle Exception?
        grid = interpret_grid(grid)
    # elif filename.endswith('.py'):
    #     with open(filename, 'r') as f:
    #         grid = eval(f.read(), {'hyperopt': hyperopt, 'hp': hp, 'scope': scope}, {})
    #         # No need to interpret the grid here.
    else:
        raise ValueError(f"File extension is not supported.")

    return grid


def interpret_grid(raw_grid, base_key=''):
    """
    Takes a raw grid, expressed as a dictionary which was loaded from a yaml or json file, and
    adds hyperopt expressions to it
    :param raw_grid:
    :return:
    """
    grid = {}
    for key in raw_grid:
        if isinstance(raw_grid[key], list):
            grid[key] = hp.choice(f"{base_key}{key}", raw_grid[key])
        elif isinstance(raw_grid[key], dict):
            grid[key] = interpret_grid(raw_grid[key], base_key=f'{key}.')
        else:
            grid[key] = raw_grid[key]
    return grid


def unit_run(program_path, hyperparams):
    """
    Runs a given hyperparametrization of the program, and returns the
    validation loss (and other metrics).
    :param program_path:
    :param hyperparams:
    :return:
    """
    hp_string = json.dumps(hyperparams)
    cp = subprocess.run(["python", program_path, "--hp", hp_string], capture_output=True, encoding="utf-8")

    # Find the metrics
    # TODO: Capture the additional metrics too
    data = re.findall(r"^---$([\s\S]+?)(?=^---$)", cp.stdout, flags=re.MULTILINE)

    # Let's find the last occurrence of the validation loss
    loss = None
    merged_dict = {}
    for item_str in data:
        try:
            item = yaml.safe_load(item_str)
            if 'Validation loss' not in item:
                continue
            loss = float(item['Validation loss'])
            if math.isnan(loss):
                loss = float('+inf')
            merged_dict = {**merged_dict, **item}  # Shallow merge
        except yaml.YAMLError:
            pass
        except ValueError:
            pass

    if loss is None:
        # No loss found in the output; abort
        print(cp.stdout)
        print(cp.stderr)
        assert False, "No loss"  # FIXME Safe fail

    measurements = {}

    if 'Best epoch' in merged_dict:
        measurements['epochs'] = int(merged_dict['Best epoch'])

    return loss, measurements


def get_computation_wrapper(program_path, repeat=1, break_on_infnan=True):
    def opt_wrapper(hh):
        # perf, measurements = unit_run(program_path, hh)
        # tune.report(loss=perf, hp=hh, measurements=measurements)
        perf = []
        measurements = []
        for i in range(repeat):
            p, m = unit_run(program_path, hh)
            perf.append(p)
            measurements.append(m)
            if break_on_infnan and not np.isfinite(p):
                break  # Break but only after the invalid value has been put in the list
        tune.report(loss=np.mean(perf), hp=hh, measurements=measurements)
    return opt_wrapper


def parallel_optim(program_path, opt_space, n_searches=1, n_repetitions=10):
    algo = HyperOptSearch(opt_space, metric="loss", mode="min")
    #re_algo = Repeater(algo, repeat=n_repetitions)

    name = os.path.basename(program_path)
    if name.endswith('.py'):
        name = name[:-3]

    analysis = tune.run(get_computation_wrapper(program_path, repeat=n_repetitions),
                        name=name, search_alg=algo, num_samples=n_searches,
                        resources_per_trial={'cpu': 10, 'gpu': 0.5})

    best_trial = analysis.get_best_trial(metric="loss", mode="min")

    #print(best_trial.__dict__)
    measurements = best_trial.last_result['measurements']
    #best_loss = best_trial.last_result['loss']
    best_hyperparams = best_trial.config

    return best_hyperparams, measurements, best_trial


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--searches', type=int, default=20)
    argparser.add_argument('--repeats', type=int, default=1)
    argparser.add_argument('--cpus', type=int, default=50)
    argparser.add_argument('--gpus', type=int, default=2)
    argparser.add_argument('grid_file', metavar='grid_file', type=str)
    argparser.add_argument('program', metavar='program', type=str)
    args = argparser.parse_args()

    print("You have invoked this script as:")
    print(sys.argv)
    print()

    ray.init(num_cpus=args.cpus, num_gpus=args.gpus)

    grid = load_grid(args.grid_file)
    program = os.path.join(os.getcwd(), args.program)  # FIXME Allow extra args
    best, measurements, best_trial = parallel_optim(program, grid, n_searches=args.searches, n_repetitions=args.repeats)
    print(best)
    print(measurements)
    print(best_trial)


if __name__ == "__main__":
    main()
