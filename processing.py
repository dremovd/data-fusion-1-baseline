import numpy as np
import pandas as pd


def load_dataset(dataset_name, drop_unlabeled=True):
    data = pd.read_parquet(dataset_name)
    data['weight'] = 1

    receipt_item_count = data.groupby('receipt_id').agg(
        {'weight': 'count'}).rename({'weight': 'receipt_item_count'}, axis=1)
    data = data.merge(receipt_item_count, on='receipt_id')
    if drop_unlabeled:
        data = data[data.category_id != -1]
    return data


def unique_item_name(data):
    grouping = {
        'receipt_dayofweek': 'first',
        'item_nds_rate': 'first',
        'receipt_item_count': 'first',
        'item_quantity': 'first',
        'item_price': 'first',
        'receipt_id': 'first',
    }
    if 'category_id' in data.columns:
        grouping['category_id'] = 'first'

    data_unique = data.groupby(['item_name']).agg(grouping).reset_index()
    return data_unique

