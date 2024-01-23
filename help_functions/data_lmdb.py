import os
import re
import lmdb
import pickle
from functools import lru_cache
import json
from copy import deepcopy


def getcid_from_name(name):
    match = re.search(r'(\d+)', name)
    assert match is not None
    return match.group(1)


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

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data

# Write all results into LMDB
env_new = lmdb.open(
    '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/3d-pubchem.lmdb',
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_new.begin(write=True)

dataset_path = '/data/lish/3D-MoLM/MolChat/data/mola-d-v3/'
for subset in ['pretrain/', 'train/', 'valid/', 'test/']:
    subset_path = dataset_path + subset
    lmdb_dataset = LMDBDataset(subset_path+'molecule_3d_feat.lmdb')

    text_name_list = os.listdir(subset_path + 'text/')
    text_name_list.sort()

    enriched_text_name_list = os.listdir(subset_path + 'polished_text/')
    enriched_text_name_list.sort()

    smiles_name_list = os.listdir(subset_path + 'smiles/')
    smiles_name_list.sort()

    ## filtering dataset that are errors
    with open(os.path.join(subset_path, 'error_ids.txt'), 'r') as f:
        error_ids = set(json.load(f))

    ## make an mapping from new id to original id.
    ## This is because the dataset is filtered, so the index of the dataset is not continuous
    mapping = {}
    for i in range(len(text_name_list)):
        if i not in error_ids:
            mapping[len(mapping)] = i

    text_name_list = [data for i, data in enumerate(text_name_list) if i not in error_ids]
    smiles_name_list = [data for i, data in enumerate(smiles_name_list) if i not in error_ids]

    for index, smiles in enumerate(smiles_name_list):
        ## load 3d
        data_loaded = lmdb_dataset[mapping[index]]

        data = deepcopy(data_loaded)
        ## load cid
        cid = getcid_from_name(smiles)

        ## load smiles
        smiles_path = os.path.join(subset_path, 'smiles', smiles_name_list[index])
        with open(smiles_path, 'r') as f:
            loaded_smiles = f.readline().strip()

        assert loaded_smiles == data['smi']

        data['smiles'] = data.pop('smi')

        ## load texts
        text_path = os.path.join(subset_path, 'text', text_name_list[index])
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        text = ' '.join(lines) + '\n'

        ## load enriched texts
        text_path = os.path.join(subset_path, 'polished_text', text_name_list[index])
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
        enriched_text = ' '.join(lines) + '\n'

        data['description'] = text
        data['enriched_description'] = enriched_text
        data['cid'] = cid
        txn_write.put(cid.encode(), pickle.dumps(data, protocol=-1))

txn_write.commit()
env_new.close()
