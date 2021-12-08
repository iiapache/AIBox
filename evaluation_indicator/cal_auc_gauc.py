import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

def cal_auc_by_sklearn(triple_list):
    pred_list = [triple[1] for triple in triple_list]
    label_list = [triple[2] for triple in triple_list]
    return roc_auc_score(label_list, pred_list)

def cal_auc_manually(triple_list):
    triple_list.sort(key=lambda x: x[1])
    rank = 0
    num_pos = 0
    num_neg = 0
    sum_rank = 0
    for (_, pred, label) in triple_list:
        rank += 1
        if label > 0:
            num_pos += 1
            sum_rank += rank
        else:
            num_neg += 1
    auc = (sum_rank - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return auc

def cal_gauc(triple_list):
    triple_list.sort(key=lambda x: (x[0], x[1]))
    pos_num, neg_num = 0, 0
    rank, pre_group_flag = 0, ""
    sum_vv, sum_pos_rank, sum_auc = 0, 0, 0.0
    valid_group, all_group = 0, 0

    for (group_flag, pred, label) in triple_list:
        if pre_group_flag == "" or group_flag != pre_group_flag:
            if pos_num > 0 and neg_num > 0 and (pos_num + neg_num >= 5) and (pos_num + neg_num <= 300):
                auc = (sum_pos_rank - pos_num * (pos_num + 1) / 2.0) / (pos_num * neg_num)
                sum_auc += auc * rank
                sum_vv += rank
                valid_group += 1
            pre_group_flag = group_flag
            rank, sum_pos_rank, pos_num, neg_num = 0, 0, 0, 0
            all_group += 1
        rank += 1
        if int(label) > 0:
            pos_num += 1
            sum_pos_rank += rank
        else:
            neg_num += 1
    print("valid_group: {}, all_group: {}".format(valid_group, all_group))
    try:
        gauc = sum_auc / sum_vv
        return gauc
    except ZeroDivisionError:
        print("Zero Error !!!")

def cal_cross_entropy(triple_list):
    sum_pred = 0
    num_pred = len(triple_list)
    ne_sum = 0
    for (_, pred, label) in triple_list:
        if pred == 0 or pred == 1:
            continue
        ne_sum += label * np.log(pred) + (1-label) * np.log(1 - pred)
        sum_pred += pred
    avg_pred = sum_pred / num_pred
    ne_sum = ne_sum / num_pred
    ne = ne_sum / (avg_pred * np.log(avg_pred) + (1 - avg_pred) * np.log(1 - avg_pred))
    return ne

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date')
    parser.add_argument('modelid')
    parser.add_argument('servingid')
    parser.add_argument('data_path')
    parser.add_argument('result_path')
    args = parser.parse_args()
    input_path = "{}/{}_{}".format(args.data_path, args.modelid, args.date)

    triple_list = []

    with open(input_path) as input_stream:
        for line in input_stream:
            input_rows = line.strip().split(" ", -1)
            triple_list.append((input_rows[0], float(input_rows[1]), float(input_rows[2])))     # group_flag, pred, label
    cal_gauc(triple_list)