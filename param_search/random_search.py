#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random

param_range = {
        'ctr_alpha': range(1, 10, 1),
        'ctr_beta': range(1, 10, 1),
        'cvr_alpha': range(1, 10, 1),
        'cvr_beta': range(1, 10, 1)
    }

print(random.randrange(10,30,2))
random.choice(range(10, 30, 2))