#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from cal_auc_gauc import cal_auc_manually, cal_gauc

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

samples = []
def merge_score(record, *args):
    score = 0.0
    for i in range(len(record)):
        score += math.pow((1.0 + args[2 * i] * record[i]), args[2 * i + 1])
    return score

def surrogate_function(*args):
    triplet_list = [("", merge_score(record, args), record[-1]) for record in samples]
    gauc = cal_gauc(triplet_list)
    return gauc

def bayesian_optimize():
    # Bounded region of parameter space
    pbounds = {'x': (2, 4), 'y': (-3, 3)}
    # pbounds = {
    #     'alpha1': (2, 4),
    #     'beta1': (-3, 3),
    #     'alpha2': (2, 4),
    #     'beta2': (-3, 3),
    # }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )
    logger = JSONLogger(path="./logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(
        init_points=2,
        n_iter=3,
    )
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)

