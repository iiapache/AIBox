#!/usr/bin/python
# coding=utf-8

import numpy
import random
import scipy.special as special
import math
from math import log


class BayesianSmoother(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, sample_num, imp_upperbound):
        ctr_list = numpy.random.beta(alpha, beta, sample_num)
        imp_list = []
        click_list = []
        for ctr in ctr_list:
            imp = random.random() * imp_upperbound
            click = imp * ctr
            imp_list.append(imp)
            click_list.append(click)
        return imp_list, click_list

    def update_using_fpi(self, imp_list, click_list, iter_num, epsilon):
        '''
            estimate alpha, beta using fixed point iteration
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imp_list, click_list, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imp_list, click_list, alpha, beta):
        '''
            fixed point iteration
        '''
        sum_alpha = 0.0
        sum_beta = 0.0
        sum_denominator = 0.0
        for i in range(len(imp_list)):
            if imp_list[i] >= click_list[i]:         # filter noise
                sum_alpha += (special.digamma(click_list[i] + alpha) - special.digamma(alpha))
                sum_beta += (special.digamma(imp_list[i] - click_list[i] + beta) - special.digamma(beta))
                sum_denominator += (special.digamma(imp_list[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (sum_alpha / sum_denominator), beta * (sum_beta / sum_denominator)

    def update_using_me(self, imp_list, click_list):
        '''
            estimate alpha, beta using moment estimation
        '''
        mean, var = self.__compute_moment(imp_list, click_list)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def __compute_moment(self, imp_list, click_list):
        '''
            using moment estimation
        '''
        ctr_list = []
        var = 0.0
        for i in range(len(imp_list)):
            if imp_list[i] > 0.0:
                ctr_list.append(float(click_list[i]) / imp_list[i])
        mean = sum(ctr_list) / len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr - mean, 2)

        return mean, var / (len(ctr_list) - 1)

def test():
    bayesian_smoother = BayesianSmoother(1, 1)
    imp_list, click_list = bayesian_smoother.sample_from_beta(10, 1000, 10000, 1000)

    bayesian_smoother.update_using_fpi(imp_list, click_list, 1000, 0.00000001)
    print(bayesian_smoother.alpha, bayesian_smoother.beta)

    bayesian_smoother.update_using_me(imp_list, click_list)
    print(bayesian_smoother.alpha, bayesian_smoother.beta)

if __name__ == "__main__":
    test()