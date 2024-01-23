import os
import re
import lmdb
import pickle
from functools import lru_cache
import torch
import numpy as np
import json
from copy import deepcopy


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, id):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(id.encode())
        data = pickle.loads(datapoint_pickled)
        return data

dataset_path = '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/'
lmdb_dataset = LMDBDataset(dataset_path+'3d-pubchem.lmdb')

permutation = torch.randperm(len(lmdb_dataset)).numpy()

# 创建一个空列表来保存满足条件的字符串的索引
pretrain_cid = []
downstream_cid = []

# 遍历列表中的每个元素
for i in permutation:
    # 使用split方法将字符串分割成单词，然后使用len函数计算单词的数量
    num_words = len(lmdb_dataset[lmdb_dataset._keys[i].decode()]['description'].split())
    # 如果单词的数量超过20，将索引添加到indices列表中
    if num_words >= 20 and len(downstream_cid) < 15000:
        downstream_cid.append(lmdb_dataset[lmdb_dataset._keys[i].decode()]['cid'])
    else:
        pretrain_cid.append(lmdb_dataset[lmdb_dataset._keys[i].decode()]['cid'])

# pretrain
env_pretrain = lmdb.open(
    '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb',
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_pretrain.begin(write=True)
for cid in pretrain_cid:
    txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
txn_write.commit()
env_pretrain.close()

# train
env_train = lmdb.open(
    '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/train/3d-pubchem.lmdb',
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_train.begin(write=True)
for cid in downstream_cid[:12000]:
    txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
txn_write.commit()
env_train.close()

# valid
env_valid = lmdb.open(
    '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/valid/3d-pubchem.lmdb',
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_valid.begin(write=True)
for cid in downstream_cid[12000:13000]:
    txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
txn_write.commit()
env_valid.close()

# test
env_test = lmdb.open(
    '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/test/3d-pubchem.lmdb',
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_test.begin(write=True)
for cid in downstream_cid[13000:]:
    txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
txn_write.commit()
env_test.close()
