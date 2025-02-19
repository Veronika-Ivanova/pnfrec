"""
Metrics.
"""
import numpy as np
import pandas as pd
import torch
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, recall_at_k
from replay.metrics import OfflineMetrics, HitRate, MRR, NDCG, Coverage
from tqdm.auto import tqdm


def compute_replay_metrics(last_item_pos, last_item_neg, preds_pos, preds_neg, train_full, relevance_col,
                    relevance_threshold, k=10):
    
    metrics = [NDCG(k), MRR(k), HitRate(k)]
    
    positive_metrics = OfflineMetrics(
        metrics, query_column='user_id', item_column='item_id', rating_column='prediction'
    )(preds_pos, last_item_pos)
    
    negative_metrics = OfflineMetrics(
        metrics, query_column='user_id', item_column='item_id', rating_column='prediction'
    )(preds_neg, last_item_neg)
    
    coverage = OfflineMetrics(
        [Coverage(k)], query_column='user_id', item_column='item_id', rating_column='prediction'
    )(pd.concat([preds_pos, preds_neg]), last_item_neg, train_full)
    
    metrics_dict = {
        'NDCG_p': round(positive_metrics['NDCG@10'], 6),
        'MRR_p': round(positive_metrics['MRR@10'], 6),
        'HR_p': round(positive_metrics['HitRate@10'], 6),
        'NDCG_n': round(negative_metrics['NDCG@10'], 6),
        'MRR_n': round(negative_metrics['MRR@10'], 6),
        'HR_n': round(negative_metrics['HitRate@10'], 6),
        'NDCG_diff': round(positive_metrics['NDCG@10'] - negative_metrics['NDCG@10'], 6),
        'MRR_diff': round(positive_metrics['MRR@10'] - negative_metrics['MRR@10'], 6),
        'HR_diff': round(positive_metrics['HitRate@10'] - negative_metrics['HitRate@10'], 6),
        'Coverage': round(coverage['Coverage@10'], 6)
    }
    
    return metrics_dict


# fix positive / negative ground tr

def compute_metrics(last_item_pos,
    last_item_neg,
    preds_pos,
    preds_neg, train_full, relevance_col,
                    relevance_threshold, k=10):
    
    # when we have 1 true positive, HitRate == Recall and MRR == MAP
    metrics_dict = {
        f'NDCG_p': round(ndcg_at_k(last_item_pos, preds_pos, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'HR_p': round(recall_at_k(last_item_pos, preds_pos, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'MRR_p': round(map_at_k(last_item_pos, preds_pos, col_user='user_id', col_item='item_id',
                           col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'NDCG_n': round(ndcg_at_k(last_item_neg, preds_neg, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'HR_n': round(recall_at_k(last_item_neg, preds_neg, col_user='user_id', col_item='item_id',
                             col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'MRR_n': round(map_at_k(last_item_neg, preds_neg, col_user='user_id', col_item='item_id',
                           col_prediction='prediction', col_rating=relevance_col, k=k), 6),
        f'Coverage': round(pd.concat([preds_pos, preds_neg]).item_id.nunique() / train_full.item_id.nunique(), 6),
    }
    
    metrics_dict['HR_diff'] = round(metrics_dict['HR_p'] - metrics_dict['HR_n'], 6)
    metrics_dict['MRR_diff'] = round(metrics_dict['MRR_p'] - metrics_dict['MRR_n'], 6)
    metrics_dict['NDCG_diff'] = round(metrics_dict['NDCG_p'] - metrics_dict['NDCG_n'], 6)

    return metrics_dict


def positive_negative_metrics(recs, last_item, 
                              relevance_col,
                              relevance_threshold,
                              train=None,
                              user_col='user_id',
                              item_col='item_id'):
    recs_list = (recs
                 .rename(columns={'user_id': user_col})
                 .groupby(user_col)[item_col]
                 .agg(lambda x: list(x))
                 .reset_index()
                 .rename(columns={item_col: 'item_ids'}))
    last_item_pos = last_item[last_item[relevance_col] >= relevance_threshold]
    last_item_neg = last_item[last_item[relevance_col] < relevance_threshold]
    tp, fn = confusion_matrix_metrics(last_item_pos, recs_list, user_col, item_col)
    fp, tn = confusion_matrix_metrics(last_item_neg, recs_list, user_col, item_col)
    
    metrics_dict = {'precision': precision(tp, fp),
                    'recall': recall(tp, fn),
                    'MCC': mcc(tp, tn, fp, fn),
                    'HR_p': hr(last_item_pos, recs_list, user_col, item_col),
                    'MRR_p': mrr(last_item_pos, recs_list, user_col, item_col),
                    'NDCG_p': ndcg(last_item_pos, recs_list, user_col, item_col),
                    'HR_n': hr(last_item_neg, recs_list, user_col, item_col),
                    'MRR_n': mrr(last_item_neg, recs_list, user_col, item_col),
                    'NDCG_n': ndcg(last_item_neg, recs_list, user_col, item_col)}
        
    metrics_dict['HR_diff'] = round(metrics_dict['HR_p'] - metrics_dict['HR_n'], 6)
    metrics_dict['MRR_diff'] = round(metrics_dict['MRR_p'] - metrics_dict['MRR_n'], 6)
    metrics_dict['NDCG_diff'] = round(metrics_dict['NDCG_p'] - metrics_dict['NDCG_n'], 6)
    
    if train is not None:
        metrics_dict['coverage'] = coverage(recs, train, item_col)
    return metrics_dict

def hr(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id',
) -> float:
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    hr_values = []
    for _, row in df.iterrows():
        hr_values.append(int(row[item_col] in row['item_ids']))
    return round(np.mean(hr_values), 6)

def mrr(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id'
) -> float:
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    mrr_values = []
    for _, row in df.iterrows():
        try:
            user_mrr = 1 / (row['item_ids'].index(row[item_col]) + 1)
        except ValueError:
            user_mrr = 0
        mrr_values.append(user_mrr)
    return round(np.mean(mrr_values), 6)

def ndcg(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id'
) -> float:
    # ideal dcg == 1 при стратегии разделения leave-one-out
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    ndcg_values = []
    for _, row in df.iterrows():
        try:
            user_ndcg = 1 / np.log2(row['item_ids'].index(row[item_col]) + 2)
        except ValueError:
            user_ndcg = 0
        ndcg_values.append(user_ndcg)
    return round(np.mean(ndcg_values), 6)

def coverage(recs: pd.DataFrame, train: pd.DataFrame, item_col='item_id') -> float:
    return round(recs[item_col].nunique() / train[item_col].nunique(), 6)

def confusion_matrix_metrics(
    ground_truth: pd.DataFrame,
    recs_list: pd.DataFrame,
    user_col='user_id',
    item_col='item_id',
) -> float:
    '''
    tp, fn for positive ground_truth
    fp, tn for negative ground_truth
    '''
    df = ground_truth.merge(recs_list, on=user_col, how='inner')
    positive_values = []
    negative_values = []
    for _, row in df.iterrows():
        positive_values.append(int(row[item_col] in row['item_ids']))
        negative_values.append(int(row[item_col] not in row['item_ids']))
    return np.sum(positive_values), np.sum(negative_values)

def mcc(tp, tn, fp, fn): 
    return round((tp * tn - fp * fn) / ((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)) ** 0.5, 6)

def precision(tp, fp):
    return round(tp / (tp + fp), 6)

def recall(tp, fn):
    return round(tp / (tp + fn), 6)
