# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from data_provider.mol_dataset import MolDataset
from torch.utils.data import DataLoader
from data_provider.unimol_dataset import D3Collater
from torch_geometric.loader.dataloader import Collater
from unicore.data import Dictionary


class MyCollater:
    def __init__(self, tokenizer, text_max_len, pad_idx, load_3d=False):
        self.pad_idx = pad_idx
        self.load_3d = load_3d
        self.d3_collater = D3Collater(pad_idx)
        self.d2_collater = Collater([], [])
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len

    def __call__(self, batch):
        if self.load_3d:
            d3_batch, text_batch = zip(*batch)
            padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(d3_batch)
            text_tokens = self.tokenizer(text_batch,
                                         truncation=True,
                                         padding='max_length',
                                         add_special_tokens=True,
                                         max_length=self.text_max_len,
                                         return_tensors='pt',
                                         return_attention_mask=True, 
                                         return_token_type_ids=False)
            return (padded_atom_vec, padded_dist, padded_edge_type), text_tokens
        else:
            return self.d2_collater(batch)


class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        dictionary=None,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.args = args
        if args.mode == 'pretrain':
            self.train_dataset = MolDataset(root+'/pretrain/', text_max_len, dictionary, args.unimol_max_atoms, enriched_descrption=args.enriched_descrption).shuffle()
        else:
            self.train_dataset = MolDataset(root+'/train/', text_max_len, dictionary, args.unimol_max_atoms).shuffle()
        self.val_dataset = MolDataset(root + '/valid/', text_max_len, dictionary, args.unimol_max_atoms).shuffle()
        self.val_dataset_match = MolDataset(root + '/valid/', text_max_len, dictionary, args.unimol_max_atoms).shuffle()
        self.test_dataset_match = MolDataset(root + '/test/', text_max_len, dictionary, args.unimol_max_atoms).shuffle()

        self.val_match_loader = DataLoader(self.val_dataset_match, 
                                           batch_size=self.match_batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers, 
                                           pin_memory=False, 
                                           drop_last=False, 
                                           persistent_workers=True,
                                           collate_fn=MyCollater(tokenizer, self.args.text_max_len, self.dictionary.pad(), self.args.use_3d))
        self.test_match_loader = DataLoader(self.test_dataset_match, 
                                            batch_size=self.match_batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers, 
                                            pin_memory=False, 
                                            drop_last=False, 
                                            persistent_workers=True,
                                            collate_fn=MyCollater(tokenizer, self.args.text_max_len, self.dictionary.pad(), self.args.use_3d))
    
    def load_unimol_dict(self):
        dictionary = Dictionary.load('./data_provider/unimol_dict.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        return dictionary

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.args.text_max_len, self.dictionary.pad(), self.args.use_3d)
        )

        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--root', type=str, default='data/3d-mol-dataset/pubchem')
        parser.add_argument('--text_max_len', type=int, default=256)
        return parent_parser
    