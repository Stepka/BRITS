import os
import time

import ujson as json
import numpy as np
import pandas as pd
from datetime import date

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


index_column_name = "merchant_name"
value_column_name = "spend"
time_column_name = "month"

default_path = "./"
brits_path = "./"
dataset_mame = ""


class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        # self.content = open(brits_path + 'json/json').readlines()
        #
        # indices = np.arange(len(self.content))
        # val_indices = np.random.choice(indices, len(self.content) // 5)
        #
        # self.val_indices = set(val_indices.tolist())

        self.gaps = pd.read_csv(default_path + dataset_mame)

        # accumulate the records within one month
        self.gaps['month'] = self.gaps['month'].apply(lambda x: self.to_month_bin(x))
        self.min_date = self.gaps['month'].min()
        self.max_date = self.gaps['month'].max()

        self.means = self.gaps.groupby(['merchant_name']).mean()['spend'].values
        self.stds = self.gaps.groupby(['merchant_name']).std()['spend'].values
        self.stds[self.stds == 0] = 1
        self.mins = self.gaps.groupby(['merchant_name']).min()['spend'].values
        self.maxs = self.gaps.groupby(['merchant_name']).max()['spend'].values
        self.base = self.maxs - self.mins
        self.base[self.base == 0] = 1

        self.shops = self.gaps['merchant_name'].unique().tolist()
        self.all_ids = self.gaps['unique_mem_id'].unique().astype('int64')
        self.train_ids = self.all_ids[
            np.where(self.gaps.groupby(['unique_mem_id'])['month'].nunique().values == self.max_date - self.min_date + 1)
        ]
        self.missed_data_ids = self.all_ids[
            np.where(self.gaps.groupby(['unique_mem_id'])['month'].nunique().values < self.max_date - self.min_date + 1)
        ]

        val_indices = np.random.choice(self.train_ids, len(self.train_ids) // 5)
        self.val_indices = set(val_indices.tolist())

        self.attributes = self.shops

    def __len__(self):
        return len(self.train_ids)
        # return len(self.content)

    def __getitem__(self, idx):
        # rec = json.loads(self.content[idx])
        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        id_ = self.train_ids[idx]
        # print("get {} item (id={})".format(idx, id_))
        data = self.gaps[self.gaps['unique_mem_id'] == id_]
        rec = self.parse_row(data, self.min_date, self.max_date)
        if id_ in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec

    def to_time_bin(self, x):
        h, m = map(int, x.split(':'))
        return h

    def to_month_bin(self, x):
        y, m, d = map(int, x.split('-'))
        return y * 12 + (m - 1)

    def from_month_bin(self, x):
        y = x // 12
        m = x % 12 + 1
        return date(y, m, 1).strftime("%Y-%m-%d")

    def parse_data(self, x):
        x = x.set_index(index_column_name).to_dict()[value_column_name]

        values = []

        for attr in self.attributes:
            if attr in x:
                values.append(x[attr])
            else:
                values.append(np.nan)
        return values

    def parse_delta(self, masks, dir_, num_rows, num_features):
        if dir_ == 'backward':
            masks = masks[::-1]

        deltas = []

        for h in range(num_rows):
            if h == 0:
                deltas.append(np.ones(num_features))
            else:
                deltas.append(np.ones(num_features) + (1 - masks[h]) * deltas[-1])

        return np.array(deltas)

    def parse_rec(self, values, masks, evals, eval_masks, dir_, num_rows, num_features):
        deltas = self.parse_delta(masks, dir_, num_rows, num_features)

        # only used in GRU-D
        forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

        rec = {}

        rec['values'] = np.nan_to_num(values).tolist()
        rec['masks'] = masks.astype('int32').tolist()
        # imputation ground-truth
        rec['evals'] = np.nan_to_num(evals).tolist()
        rec['eval_masks'] = eval_masks.astype('int32').tolist()
        rec['forwards'] = forwards.tolist()
        rec['deltas'] = deltas.tolist()

        return rec

    def parse_row(self, data, min_date, max_date):

        evals = []

        not_for_train = False

        # merge all the metrics within one month
        for m in range(min_date, max_date + 1):
            if len(data[data['month'] == m]) == 0:
                print("missed data for {}".format(self.from_month_bin(m)))
                not_for_train = True
            evals.append(self.parse_data(data[data['month'] == m]))

        # if user has missed data we exclude him from train dataset
        if not_for_train:
            return False

        # normalization
        # evals = (np.array(evals) - self.means) / self.stds
        evals = (np.array(evals) - self.mins) / self.base

        evals = np.array(evals)

        shp = evals.shape

        evals = evals.reshape(-1)

        # randomly eliminate 10% values as the imputation ground-truth
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, len(indices) // 10)

        values = evals.copy()
        values[indices] = np.nan

        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

        evals = evals.reshape(shp)
        values = values.reshape(shp)

        masks = masks.reshape(shp)
        eval_masks = eval_masks.reshape(shp)

        rec = {'label': 1}

        num_rows = max_date - min_date + 1

        # prepare the model for both directions
        rec['forward'] = self.parse_rec(values, masks, evals, eval_masks, 'forward',
                                   num_rows, len(self.attributes))
        rec['backward'] = self.parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], 'backward',
                                    num_rows, len(self.attributes))

        # rec = json.dumps(rec)

        # fs.write(rec + '\n')

        return rec


def collate_fn(recs):
    forward = map(lambda x: x['forward'], recs)
    backward = map(lambda x: x['backward'], recs)

    def to_tensor_dict(recs):
        values = torch.FloatTensor(map(lambda r: r['values'], recs))
        masks = torch.FloatTensor(map(lambda r: r['masks'], recs))
        deltas = torch.FloatTensor(map(lambda r: r['deltas'], recs))

        evals = torch.FloatTensor(map(lambda r: r['evals'], recs))
        eval_masks = torch.FloatTensor(map(lambda r: r['eval_masks'], recs))
        forwards = torch.FloatTensor(map(lambda r: r['forwards'], recs))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(map(lambda x: x['label'], recs))
    ret_dict['is_train'] = torch.FloatTensor(map(lambda x: x['is_train'], recs))

    return ret_dict


def get_loader(batch_size=64, shuffle=True):
    data_set = MySet()
    data_iter = DataLoader(dataset=data_set, \
                           batch_size=batch_size, \
                           num_workers=4, \
                           shuffle=shuffle, \
                           pin_memory=True, \
                           collate_fn=collate_fn
    )

    return data_iter
