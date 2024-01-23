import torch
from torch.utils.data import Dataset
import os
from data_provider.unimol_dataset import D3Dataset_cid


class MolDataset(Dataset):
    def __init__(self, root, text_max_len, unimol_dict=None, max_atoms=256, prompt='', return_prompt=False, enriched_descrption=False):
        super(MolDataset, self).__init__()
        self.prompt = prompt
        self.return_prompt = return_prompt
        self.enriched_descrption = enriched_descrption

        self.root = root
        self.text_max_len = text_max_len
        self.tokenizer = None

        target_path = os.path.join(root, '3d-pubchem.lmdb')
        self.d3_dataset = D3Dataset_cid(target_path, unimol_dict, max_atoms)
        self.cid_list = [cid.decode() for cid in self.d3_dataset.lmdb_dataset._keys]
        self.permutation = None
    
    def shuffle(self):
        ## shuffle the dataset using a permutation matrix
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return len(self.cid_list)

    def get_3d(self, index):
        atom_vec, coordinates, edge_type, dist, smiles, description, enriched_description = self.d3_dataset[self.cid_list[index]]
        if self.enriched_descrption:
            return_description = enriched_description
        else:
            return_description = description

        if self.return_prompt:
            smiles_prompt = self.prompt.format(smiles[:96])
            return (atom_vec, coordinates, edge_type, dist, smiles), smiles_prompt, return_description, index
        return (atom_vec, coordinates, edge_type, dist, smiles), return_description

    def __getitem__(self, index):
        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]
        return self.get_3d(index)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


if __name__ == '__main__':
    from unicore.data import Dictionary
    dictionary = Dictionary.load('unimol_dict.txt')
    dictionary.add_symbol("[MASK]", is_special=True)
    dataset = MolDataset('../data/3d-mol-dataset/pubchem'+'/test/', 128, dictionary, 256)

    # 初始化计数器和统计量
    total_files = 0
    word_counts = []

    for index in range(0, len(dataset)):
        total_files += 1
        (atom_vec, coordinates, edge_type, dist, smiles), description = dataset[index]
        word_count = len(description.split())
        word_counts.append(word_count)

    # 计算统计量
    average_words = sum(word_counts) / total_files if total_files else 0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0

    # 输出结果
    print(f"Total files: {total_files}")
    print(f"Average words per file: {average_words:.2f}")
    print(f"Minimum words in a file: {min_words}")
    print(f"Maximum words in a file: {max_words}")
