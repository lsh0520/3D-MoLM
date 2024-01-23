import os
import os.path as osp
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

path = '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchemqc/'
lmdb_dataset = LMDBDataset(path+'pubchemqc_database.lmdb')

ind_path = osp.join(path, 'scaffold_split_inds.json')
with open(ind_path, 'r') as f:
     inds = json.load(f)

pretrain_index = inds['train'] + inds['valid']

ind_path = osp.join(path, 'scaffold_test_split_inds.json')
with open(ind_path, 'r') as f:
     inds = json.load(f)

train_index = inds['train']
valid_index = inds['valid']
test_index = inds['test']
exit()

# # pretrain
# env_pretrain = lmdb.open(
#     '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/pretrain/3d-pubchem.lmdb',
#     subdir=False,
#     readonly=False,
#     lock=False,
#     readahead=False,
#     meminit=False,
#     max_readers=1,
#     map_size=int(100e9),
# )
# txn_write = env_pretrain.begin(write=True)
# for cid in pretrain_cid:
#     txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
# txn_write.commit()
# env_pretrain.close()
#
# # train
# env_train = lmdb.open(
#     '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/train/3d-pubchem.lmdb',
#     subdir=False,
#     readonly=False,
#     lock=False,
#     readahead=False,
#     meminit=False,
#     max_readers=1,
#     map_size=int(100e9),
# )
# txn_write = env_train.begin(write=True)
# for cid in downstream_cid[:12000]:
#     txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
# txn_write.commit()
# env_train.close()
#
# # valid
# env_valid = lmdb.open(
#     '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/valid/3d-pubchem.lmdb',
#     subdir=False,
#     readonly=False,
#     lock=False,
#     readahead=False,
#     meminit=False,
#     max_readers=1,
#     map_size=int(100e9),
# )
# txn_write = env_valid.begin(write=True)
# for cid in downstream_cid[12000:13000]:
#     txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
# txn_write.commit()
# env_valid.close()
#
# # test
# env_test = lmdb.open(
#     '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/test/3d-pubchem.lmdb',
#     subdir=False,
#     readonly=False,
#     lock=False,
#     readahead=False,
#     meminit=False,
#     max_readers=1,
#     map_size=int(100e9),
# )
# txn_write = env_test.begin(write=True)
# for cid in downstream_cid[13000:]:
#     txn_write.put(cid.encode(), pickle.dumps(lmdb_dataset[cid], protocol=-1))
# txn_write.commit()
# env_test.close()