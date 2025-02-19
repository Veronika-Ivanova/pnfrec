"""
Filter interactions.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def train_val_test_split(
    df,
    relevance_threshold,
    relevance_col,
    test_quantile=0.9,
    test_validation_ratio=0.5,
    item_min_count=5,
    user_min_count=5
):
    """
    Split clickstream by date.
    
    Split validation and test by users in `test_validation_ratio` proportion.
    """
    df = df.sort_values(['user_id', 'timestamp'])
    df['user_id'] = df['user_id'].astype('category').cat.codes
    df['item_id'] = df['item_id'].astype('category').cat.codes
    df = filter_items(df, item_min_count)
    df = filter_users(df, user_min_count)
                 
    test_timepoint = df['timestamp'].quantile(
    q=test_quantile, interpolation='nearest'
    )
    test_full = df.query('timestamp >= @test_timepoint')
    train = df.drop(test_full.index)
    
    test_full = test_full[test_full['user_id'].isin(train['user_id'])]
    test_full = test_full[test_full['item_id'].isin(train['item_id'])]
    
    users_val, users_test = train_test_split(
        test_full['user_id'].unique(),
        test_size=test_validation_ratio,
        random_state=42
    )
    
    val = test_full[test_full['user_id'].isin(users_val)]
    test = test_full[test_full['user_id'].isin(users_test)]
    
    train = add_time_idx(train)
    test = add_time_idx(test)
    val = add_time_idx(val)
    
    test.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)

    return train, val, test


def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df


def prepare_splitted_data(data_path,
                          relevance_col,
                          relevance_threshold,
                          user_col='user_id',
                          min_items_per_user=2,
                          timestamp_col='timestamp',
                          filter_negative=False):
    train = pd.read_parquet(os.path.join(data_path, 'train.parquet'))
    validation = pd.read_parquet(os.path.join(data_path, 'validation.parquet'))
    test = pd.read_parquet(os.path.join(data_path, 'test.parquet'))
    
    train.item_id = train.item_id + 1
    validation.item_id = validation.item_id + 1
    test.item_id = test.item_id + 1
    
    train = filter_users_by_history_len(train, user_col, relevance_col, relevance_threshold)
    validation = filter_users_by_history_len(validation, user_col, relevance_col, relevance_threshold)
    test = filter_users_by_history_len(test, user_col, relevance_col, relevance_threshold)
    test = test[test[user_col].isin(train[user_col])]
    validation = validation[validation[user_col].isin(train[user_col])]
    
    train_test_users = train[train[user_col].isin(test[user_col])]
    test = pd.concat([train_test_users, test])
    train_val_users = train[train[user_col].isin(validation[user_col].unique())]
    validation = pd.concat([train_val_users, validation])
    
    train_full = train.copy()
    
    if filter_negative:
        train = train[train[relevance_col] >= relevance_threshold]    

    train = add_time_idx(train, user_col=user_col)
    validation = add_time_idx(validation, user_col=user_col)
    test = add_time_idx(test, user_col=user_col)
    
    test_full_history = test.groupby(user_col)
    last_item = test_full_history.tail(1)
    test = test_full_history.head(-1)
    
    last_item_pos = last_item[last_item[relevance_col] >= relevance_threshold]
    last_item_neg = last_item[last_item[relevance_col] < relevance_threshold]
    
    test_pos = test[test[user_col].isin(last_item_pos[user_col])] 
    test_neg = test[test[user_col].isin(last_item_neg[user_col])] 
    
    return train, train_full, validation, test_pos, test_neg, last_item_pos, last_item_neg


def filter_users_by_history_len(df, 
                                user_col, 
                                relevance_col, 
                                relevance_threshold, 
                                min_items_per_user=2, 
                                filter_by_positive_items=True):
    if filter_by_positive_items:
        user_count = df[df[relevance_col] >= relevance_threshold][user_col].value_counts()
    else:
        user_count = df[user_col].value_counts()
    appropriate_users = user_count[user_count >= min_items_per_user].index
    df = df[df.loc[:, user_col].isin(appropriate_users)]
    return df


def filter_items(df, item_min_count, item_col='item_id'):

    print('Filtering items..')

    item_count = df.groupby(item_col).user_id.nunique()

    item_ids = item_count[item_count >= item_min_count].index
    print(f'Number of items before {len(item_count)}')
    print(f'Number of items after {len(item_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df.item_id.isin(item_ids)]
    print(f'Interactions length after: {len(df)}')

    return df


def filter_users(df, user_min_count, user_col='user_id'):

    print('Filtering users..')

    user_count = df.groupby(user_col).item_id.nunique()

    user_ids = user_count[user_count >= user_min_count].index
    print(f'Number of users before {len(user_count)}')
    print(f'Number of users after {len(user_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df.user_id.isin(user_ids)]
    print(f'Interactions length after: {len(df)}')

    return df
