# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.mol_dataset import MolDataset
from data_provider.unimol_dataset import D3Collater
import argparse


class TrainCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.d3_collater = D3Collater(pad_idx)
        self.mol_token_id = mol_token_id
        self.pad_idx = pad_idx

    def __call__(self, batch):
        graphs, smiles_prompt, texts, indices = zip(*batch)

        padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs)
        graph_batch = (padded_atom_vec, padded_dist, padded_edge_type)

        input_pair = [[p, t] for p, t in zip(smiles_prompt, texts)]

        self.tokenizer.padding_side = 'left'
        text_batch = self.tokenizer(input_pair,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.text_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True,
                                    return_token_type_ids=True)

        is_mol_token = text_batch.input_ids == self.mol_token_id

        assert torch.sum(is_mol_token).item() == 8 * len(batch), print(input_pair)

        text_batch['is_mol_token'] = is_mol_token
        return graph_batch, text_batch


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.d3_collater = D3Collater(pad_idx)
        self.mol_token_id = mol_token_id
        self.pad_idx = pad_idx

    def __call__(self, batch):
        graphs, smiles_prompt, texts, indices = zip(*batch)

        padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = self.d3_collater(graphs)
        graph_batch = (padded_atom_vec, padded_dist, padded_edge_type)
        
        self.tokenizer.padding_side = 'right'
        prompt_batch = self.tokenizer(smiles_prompt,
                                      truncation=False,
                                      padding='longest',
                                      add_special_tokens=True,
                                      return_tensors='pt',
                                      return_attention_mask=True,
                                      return_token_type_ids=False)

        is_mol_token = prompt_batch.input_ids == self.mol_token_id
        prompt_batch['is_mol_token'] = is_mol_token
        
        target_dict = {'targets': texts, 'indices': indices}
        return graph_batch, prompt_batch, target_dict


class Stage2DM(LightningDataModule):
    def __init__(
            self,
            mode: str = 'pretrain',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            dictionary=None,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = "Below is an instruction that describes a task, paired with an input molecule. Write a response that appropriately completes the request.\n" \
                      "Instruction: Describe the input molecule.\n" \
                      "Input molecule: {} <mol><mol><mol><mol><mol><mol><mol><mol>.\n" \
                      "Response: "
        self.dictionary = dictionary

        if self.mode == 'pretrain':
            self.train_dataset = MolDataset(root + '/pretrain/', text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True, enriched_descrption=args.enriched_descrption).shuffle()
        else:
            self.train_dataset = MolDataset(root+'/train/', text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True).shuffle()
        self.val_dataset = MolDataset(root+'/valid/', text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True).shuffle()
        self.test_dataset = MolDataset(root+'/test/', text_max_len, unimol_dict=dictionary, max_atoms=args.unimol_max_atoms, prompt=self.prompt, return_prompt=True).shuffle()
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        if hasattr(self, 'pretrain_dataset'):
            self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        self.mol_token_id = self.tokenizer.mol_token_id

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=False,
                            drop_last=True,
                            persistent_workers=True,
                            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad())
                            )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad()),
        )
        return val_loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.mol_token_id, self.dictionary.pad()),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--inference_batch_size', type=int, default=8)
        parser.add_argument('--root', type=str, default='data/3d-mol-dataset/pubchem')
        parser.add_argument('--text_max_len', type=int, default=320)
        return parent_parser


if __name__ == '__main__':
    from unicore.data import Dictionary
    from transformers import AutoTokenizer
    from tqdm import tqdm
    dictionary = Dictionary.load('unimol_dict.txt')
    dictionary.add_symbol("[MASK]", is_special=True)
    llm_model = '/data/lish/3D-MoLM/MolChat/llama-2-hf-7b'
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
    llm_tokenizer.mol_token_id = llm_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_batch_size', type=int, default=16)
    parser.add_argument("--unimol-max-atoms", type=int, default=256)
    parser.add_argument('--num_query_token', type=int, default=8)
    parser.add_argument('--llm_model', type=str, default="../llama-2-hf-7b")
    parser.add_argument('--use_3d', action='store_true', default=True)
    parser.add_argument('--enriched_descrption', action='store_true', default=True)
    args = parser.parse_args()

    dm = Stage2DMV2(mode='pretrain',
                    num_workers=2,
                    batch_size=4,
                    root='/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchem',
                    text_max_len=320,
                    dictionary=dictionary,
                    tokenizer=llm_tokenizer,
                    args=args)

    data_loader = dm.train_dataloader()
    for i, data in enumerate(tqdm(data_loader)):
        pass
