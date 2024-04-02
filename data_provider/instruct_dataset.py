import torch
import json
from torch.utils.data import Dataset
import os
import re
from data_provider.unimol_dataset import D3Dataset, D3Dataset_cid
from datasets import load_dataset, Features, Value


def getcid_from_name(name):
    match = re.search(r'(\d+)', name)
    assert match is not None
    return match.group(1)


class InstructDataset(Dataset):
    def __init__(self, root, mode, split, text_max_len, unimol_dict=None, max_atoms=256):
        super(InstructDataset, self).__init__()

        split_path = os.path.join(root, split)
        data_paths = []
        if mode.find('des') >= 0:
            data_paths.append(split_path + '/descriptive_properties.json')
        elif mode.find('2d') >= 0:
            data_paths.append(split_path + '/2d_computed_properties_unit.json')
        elif mode.find('3d') >= 0:
            data_paths.append(split_path + '/3d_computed_properties_unit.json')
        elif mode.find('train') >= 0:
            data_paths.append(split_path + '/descriptive_properties.json')
            data_paths.append(split_path + '/2d_computed_properties_unit.json')
            data_paths.append(split_path + '/3d_computed_properties_unit.json')
        else:
            raise NotImplementedError

        features = Features({"instruction": Value("string"), "input": Value("string"), "output": Value("string"), "task": Value("string")})
        self.instruct_dataset_text = load_dataset("json", data_files=data_paths, features=features)['train']

        # Instruction prompt template
        self.prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request." \
                      "\n### Instruction: {}" \
                      "\n### Input: <mol><mol><mol><mol><mol><mol><mol><mol> {}" \
                      "\n### Response: "

        self.root = root
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(split_path+'/graph/')
        self.graph_name_list.sort()
        self.tokenizer = None

        if not mode.find('3d') >= 0:
            ### load RDkit 3D raw conformations
            target_path = os.path.join(split_path, 'molecule_3d_feat.lmdb')
            self.pubchem_2d_property_dataset = D3Dataset(target_path, unimol_dict, max_atoms)

            ## filtering dataset that are errors
            with open(os.path.join(split_path, 'error_ids.txt'), 'r') as f:
                error_ids = set(json.load(f))
            ## make an mapping from new id to original id.
            ## This is because the dataset is filtered, so the index of the dataset is not continuous
            self.mapping = {}
            for i in range(len(self.graph_name_list)):
                if i not in error_ids:
                    self.mapping[len(self.mapping)] = i

            self.graph_name_list = [data for i, data in enumerate(self.graph_name_list) if i not in error_ids]
            # self.smiles_name_list = [data for i, data in enumerate(self.smiles_name_list) if i not in error_ids]
            # self.text_name_list = [data for i, data in enumerate(self.text_name_list) if i not in error_ids]

            self.cid_idx_dict = {}
            for i, text_name in enumerate(self.graph_name_list):
                self.cid_idx_dict[getcid_from_name(text_name)] = i

        if mode.find('3d') >= 0 or mode.find('train') >= 0:
            ### load PubChemQC 3D conformations
            target_path = os.path.join(root, 'molecule3d_database.lmdb')
            self.pubchem_3d_property_dataset = D3Dataset_cid(target_path, unimol_dict, max_atoms)

        self.permutation = None
    
    def __len__(self):
        return len(self.instruct_dataset_text)

    def shuffle(self):
        ## shuffle the dataset using a permutation matrix
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __getitem__(self, index):
        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]
        return self.get_3d(index)

    def get_3d(self, index):
        data = self.instruct_dataset_text[index]
        cid = data['input']
        output = data['output']
        task = data['task']
        instruction = data['instruction']

        if task in ['Description', 'Molecular Weight', 'LogP', 'Topological Polar Surface Area', 'Complexity']:
            idx = self.cid_idx_dict[cid]
            atom_vec, coordinates, edge_type, dist, smiles = self.pubchem_2d_property_dataset[self.mapping[idx]]
        elif task in ['HOMO', 'LUMO', 'HOMO-LUMO Gap', 'SCF Energy']:
            atom_vec, coordinates, edge_type, dist, smiles = self.pubchem_3d_property_dataset[cid]
        else:
            raise NotImplementedError

        if task in ['HOMO', 'LUMO', 'HOMO-LUMO Gap', 'SCF Energy', 'Molecular Weight', 'LogP', 'Topological Polar Surface Area', 'Complexity']:
            input = self.prompt.format(instruction, smiles[:64])
        else:
            input = self.prompt.format(instruction[:64], smiles[:64])

        return (atom_vec, coordinates, edge_type, dist, smiles), input, output, task


# if __name__ == '__main__':
#     from unicore.data import Dictionary
#     dictionary = Dictionary.load('unimol_dict.txt')
#     dictionary.add_symbol("[MASK]", is_special=True)
#     train_dataset = InstructDataset(root='../data/mola-d-v3/',
#                                     split='valid',
#                                     text_max_len=128,
#                                     unimol_dict=dictionary,
#                                     max_atoms=256,
#                                     prompt='Given the molecule: <mol><mol><mol><mol><mol><mol><mol><mol>{}, ')
#     a = train_dataset[0]
