import os
import numpy as np
import torch
import random
import lmdb
import pickle
from functools import lru_cache
from unicore.data import data_utils
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset


class LMDBDataset_cid:
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
    def __getitem__(self, cid):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(cid.encode())
        data = pickle.loads(datapoint_pickled)
        return data


class D3Dataset_cid(Dataset):
    def __init__(self, path, dictionary, max_atoms=256):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.lmdb_dataset = LMDBDataset_cid(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, cid):
        data = self.lmdb_dataset[cid]
        smiles = data['smiles']
        description = data['description']
        enriched_description = data['enriched_description']
        ## deal with 3d coordinates
        atoms_orig = np.array(data['atoms'])
        atoms = atoms_orig.copy()
        coordinate_set = data['coordinates']
        coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
        assert len(atoms) == len(coordinates) and len(atoms) > 0
        assert coordinates.shape[1] == 3

        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) <= self.__max_atoms

        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)
        return atom_vec, coordinates, edge_type, dist, smiles, description, enriched_description


class LMDBDataset_index:
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
        datapoint_pickled = self.env.begin().get(idx.encode())
        data = pickle.loads(datapoint_pickled)
        return data


class D3Dataset_index(Dataset):
    def __init__(self, path, dictionary, max_atoms=256):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.lmdb_dataset = LMDBDataset_index(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, index):
        data = self.lmdb_dataset[index]
        smiles = data['smi']
        ## deal with 3d coordinates
        atoms_orig = np.array(data['atoms'])
        atoms = atoms_orig.copy()
        coordinate_set = data['coordinates_list']
        coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
        assert len(atoms) == len(coordinates) and len(atoms) > 0
        assert coordinates.shape[1] == 3

        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if self.max_atoms > 0 and len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) < self.__max_atoms, print(len(atoms), atoms_orig, index)
        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)
        return atom_vec, coordinates, edge_type, dist, smiles
        

def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class D3Collater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple
    
    def __call__(self, samples):
        atom_vec, coordinates, edge_type, dist, smiles = zip(*samples)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_coordinates = collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        return padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles


if __name__ == '__main__':
    from unicore.data import Dictionary
    from torch.utils.data import DataLoader
    # split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
    path = '/data/lish/3D-MoLM/MolChat/data/mola-d-v2/molecule3d_database.lmdb'
    dictionary = Dictionary.load('/data/lish/zyliu/MolChat/data_provider/unimol_dict.txt')
    dictionary.add_symbol("[MASK]", is_special=True)
    dataset = D3Dataset_cid(path, dictionary, 256)
    pass


