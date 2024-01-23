import os
import lmdb
import pickle
import torch
import pandas as pd
import os.path as osp
import random
import numpy as np
import json
from datasets import load_dataset, Features, Value

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

    def __getitem__(self, cid):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(cid.encode())
        data = pickle.loads(datapoint_pickled)
        return data

subset_dict = {'pretrain': [],
               'train': [],
               'valid': [],
               'test': []}

path = '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/'

load_des_root = '/data/lish/3D-MoLM/MolChat/data/mola-d-v3/'
loaded_des_path = []

for subset in ['pretrain', 'train', 'valid', 'test']:
    loaded_des_path.append(load_des_root + subset + '/descriptive_properties.json')
    lmdb_dataset = LMDBDataset(path + subset + '/3d-pubchem.lmdb')
    for i in range(len(lmdb_dataset)):
        cid = lmdb_dataset[lmdb_dataset._keys[i].decode()]['cid']
        subset_dict[subset].append(cid)
    subset_dict[subset] = set(subset_dict[subset])

features = Features({"instruction": Value("string"), "input": Value("string"), "output": Value("string"), "task": Value("string")})
des_dataset_text = load_dataset("json", data_files=loaded_des_path, features=features)['train']

new_subset_dict = {'pretrain': [],
                   'train': [],
                   'valid': [],
                   'test': []}

for data in des_dataset_text:
    cid = data['input']
    for subset in ['pretrain', 'train', 'valid', 'test']:
        if cid in subset_dict[subset]:
            new_subset_dict[subset].append(data)
            break

for subset in ['pretrain', 'train', 'valid', 'test']:
    json.dump(new_subset_dict[subset], open(path + subset + '/2d_descriptive_properties.json', 'w'), indent=4)
