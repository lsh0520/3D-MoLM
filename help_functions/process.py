from pathlib import Path
import lmdb
import gzip
from rdkit import Chem
import re
import pickle
from tqdm import tqdm
import pubchempy as pcp


def getcid_from_name(name):
    regex = re.compile(r'PUBCHEM (\d+) B3LYP 6-31G\(d\) gamess geometry optimization')
    match = regex.match(name)
    assert match is not None
    return match.group(1)


def read_sdf_gz_and_write_to_lmdb(file_path, lmdb_env, counter):
    with lmdb_env.begin(write=True) as txn:
        with open(file_path, 'rb') as f:
            supplier = Chem.ForwardSDMolSupplier(f)
            for molecule in tqdm(supplier):
                if molecule is None:
                    print(file_path)
                    exit()
                name = molecule.GetProp('_Name')
                cid = str(int(getcid_from_name(name)))
                
                # Get 3D Conformer if available
                conformers = []
                num_conformers = molecule.GetNumConformers()
                for i in range(num_conformers):
                    conformer = molecule.GetConformer(i)
                    conformers.append(conformer.GetPositions())

                # Get SMILES
                smiles = Chem.MolToSmiles(molecule)
                atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
                data = {
                    'cid': cid,
                    'coordinates_list': conformers,
                    'smi': smiles,
                    'atoms': atoms
                }
                data = pickle.dumps(data)

                txn.put(str(counter).encode(), data)
                counter = counter + 1
    return counter


env = lmdb.open('/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchemqc/pubchemqc_database.lmdb', map_size=int(100e9), subdir=False, lock=False,)
file_list = ['combined_mols_0_to_1000000.sdf', 'combined_mols_1000000_to_2000000.sdf', 'combined_mols_2000000_to_3000000.sdf', 'combined_mols_3000000_to_3899647.sdf']

counter = 0
for file in tqdm(file_list):
    path = Path('/data/lish/3D-MoLM/MolChat/data/3d-mol-dataset/pubchemqc') / file
    counter = read_sdf_gz_and_write_to_lmdb(path, env, counter)
    print(counter)
    