#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def calDCG(score_list):
    '''
    calculate the DCG
    :param score_list: labeled real score list of ranked items.
    :return:
    '''
    score_nparray = np.asarray(score_list, dtype=np.float32)
    return np.sum(
        np.divide(score_nparray, np.log2(np.arange(score_nparray.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def calNDCG(rank_list, label_list, k):
    '''
    calculate the nDCG@k
    :param rank_list:
    :param label_list:
    :param k:
    :return:
    '''
    k = min([k, len(rank_list), len(label_list)])
    it2rel = {item: rel for (item, rel) in label_list}
    rank_score_list = [it2rel.get(item, 0.0) for item in rank_list[:k]]
    ideal_score_list = [rel for (_, rel) in label_list[:k]]

    idcg = calDCG(ideal_score_list)
    dcg = calDCG(rank_score_list)

    if dcg == 0.0 or idcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

if __name__ == "__main__":
    rank_list = [1, 4, 5]   # itemid
    label_list = [(1, 3.0), (2, 2.0), (3, 1.0)]       # (itemid, rel) sorted by relevance in desc order.
    k = 3
    ndcg = calNDCG(rank_list, label_list, k)
    print("ndcg: {}".format(ndcg))
    rank_score_list = [1.0, 3.0, 2.0]
    dcg = calDCG(rank_score_list)
    print("dcg: {}".format(dcg))