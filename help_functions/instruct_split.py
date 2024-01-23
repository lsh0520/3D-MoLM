import os
import lmdb
import pickle
import torch
import pandas as pd
import os.path as osp
import random
import numpy as np
import json

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

path = '/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem/'
data_df = pd.read_csv(path + '2d_computed_properties.csv')

instruction_templates = [
            "What is the [property] of this molecule? ",
            "Could you give me the [property] value of this molecule? ",
            "I would like to know the [property] of this molecule, can you provide it? ",
            "Please provide the [property] value for this molecule. ",
            "I am interested in the [property] of this molecule, could you tell me what it is? ",
            "I would like to know the [property] of this molecule, could you please provide it? ",
            "I need to know the [property] of this molecule, could you please provide it? ",
            "Please provide me with the [property] value of this molecule. "
            "Determine the [property] value of this molecule. "
            ]

further_instruction = "If uncertain, provide an estimate. Respond with the numerical value only."

output_template = "The [property] for the input molecule is [output]."

properties = [
            "Molecular Weight",
            "LogP",
            "Topological Polar Surface Area",
            "Complexity",
            ]

instruction_list = []
for subset in ['pretrain', 'train', 'valid', 'test']:
    lmdb_dataset = LMDBDataset(path + subset + '/3d-pubchem.lmdb')
    for i in range(len(lmdb_dataset)):
        cid = lmdb_dataset[lmdb_dataset._keys[i].decode()]['cid']
        row = data_df[data_df['cid'] == int(cid)]
        for property_ in properties:
            output = row[property_].item()
            if not np.isnan(output):
                output = "{:.2f}".format(output)
                if property_ == "Molecular Weight":
                    output = output + ' g/mol'
                elif property_ == "Topological Polar Surface Area":
                    output = output + ' Å²'
                final_output = output_template.replace('[property]', property_)
                final_output = final_output.replace('[output]', output)
                for template in random.sample(instruction_templates, 1):
                    instruction = template.replace('[property]', property_)
                    instruction = instruction + further_instruction
                    instruction_list.append({"instruction": instruction,
                                             "input": cid,
                                             "output": final_output,
                                             "task": property_, })
    json.dump(instruction_list, open(path + subset + '/2d_computed_properties.json', 'w'), indent=4)
    instruction_list = []

