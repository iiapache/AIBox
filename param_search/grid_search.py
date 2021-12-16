#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.model_selection import ParameterGrid

param_range = {
        'ctr_alpha': range(1, 10, 1),
        'ctr_beta': range(1, 10, 1),
        'cvr_alpha': range(1, 10, 1),
        'cvr_beta': range(1, 10, 1)
    }

for param in ParameterGrid(param_range):
    print(param)