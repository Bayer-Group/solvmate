import argparse
from pathlib import Path
from dataset import GraphDataset
from rdkit import Chem
from rdkit.Chem import SDWriter
import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset

from dataset import GraphDataset
from util import collate_reaction_graphs
from model import nmrMPNN, training, inference

from sklearn.metrics import mean_absolute_error,r2_score
from scipy import stats




def compile_npz_file(sd_file):
    # the dataset can be downloaded from
    # https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help
    # nmrshiftdb2withsignals.sd

    def get_atom_shifts(mol):
        
        molprops = mol.GetPropsAsDict()
        
        atom_shifts = {}
        for key in molprops.keys():
        
            if key.startswith('Spectrum 13C'):
                
                for shift in molprops[key].split('|')[:-1]:
                
                    [shift_val, _, shift_idx] = shift.split(';')
                    shift_val, shift_idx = float(shift_val), int(shift_idx)
                
                    if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []
                    atom_shifts[shift_idx].append(shift_val)

        return atom_shifts


    def add_mol(mol_dict, mol):

        def _DA(mol):

            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
            
            return D_list, A_list

        def _chirality(atom):

            if atom.HasProp('Chirality'):
                #assert atom.GetProp('Chirality') in ['Tet_CW', 'Tet_CCW']
                c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
            else:
                c_list = [0, 0]

            return c_list

        def _stereochemistry(bond):

            if bond.HasProp('Stereochemistry'):
                #assert bond.GetProp('Stereochemistry') in ['Bond_Cis', 'Bond_Trans']
                s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
            else:
                s_list = [0, 0]

            return s_list    
            

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2

        D_list, A_list = _DA(mol)
        rings = mol.GetRingInfo().AtomRings()
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

        shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
        mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)

        mol_dict['shift'].append(shift)
        mol_dict['mask'].append(mask)
        mol_dict['smi'].append(Chem.MolToSmiles(mol))
        
        if n_edge > 0:

            bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
            bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]   
            
            edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
            edge_attr = np.vstack([edge_attr, edge_attr])
            
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
            src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
            dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
            
            mol_dict['edge_attr'].append(edge_attr)
            mol_dict['src'].append(src)
            mol_dict['dst'].append(dst)
        
        return mol_dict


    molsuppl = Chem.SDMolSupplier(str(sd_file), removeHs = False)

    atom_list = ['H', 'Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi']
    charge_list = [1, 2, 3, -1, -2, -3, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]

    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

    rdBase.DisableLog('rdApp.error') 
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

    mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'shift': [],
                'mask': [],
                'smi': []}
                    
    for i, mol in enumerate(molsuppl):

        try:
            Chem.SanitizeMol(mol)
            si = Chem.FindPotentialStereo(mol)
            #import pdb; pdb.set_trace()
            mol_probe = Chem.AddHs(mol)
            atom_types = {atm.GetSymbol() for atm in mol_probe.GetAtoms()}
            if 'H' not in atom_types and 'C' not in atom_types:
                print("skipping smiles",Chem.MolToSmiles(mol_probe))
                continue
            for element in si:
                if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                    mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                    mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
            assert '.' not in Chem.MolToSmiles(mol)
        except:
            raise

        for j, atom in enumerate(mol.GetAtoms()):
            # Idea: because we are at predict time, every atom is
            # relevant, but we don't have a true value so set it to zero...
            atom.SetDoubleProp('shift', 0)
            atom.SetBoolProp('mask', 1)

        mol = Chem.RemoveHs(mol)
        mol_dict = add_mol(mol_dict, mol)

        if (i+1) % 1000 == 0: print('%d/%d processed' %(i+1, len(molsuppl)))

    print('%d/%d processed' %(i+1, len(molsuppl)))   

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    mol_dict['shift'] = np.hstack(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])

    for key in mol_dict.keys(): 
        print(key, mol_dict[key].shape, mol_dict[key].dtype)
        
    out_file = f'./{sd_file.name}.npz'
    np.savez_compressed(out_file, data = [mol_dict])
    return out_file

from smal.all import random_fle
def run_for_smiles(smis:list[str],):

    sd_file = Path(random_fle("sdf"))

    # Create an SD writer
    sd_writer = SDWriter(str(sd_file))

    # Write molecules to the SD file
    for idx, smiles in enumerate(smis):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol.SetProp('_Name', f'Molecule_{idx + 1}')
            sd_writer.write(mol)

    # Close the SD writer
    sd_writer.close()

    npz_file = compile_npz_file(sd_file)
    data_pred = GraphDataset(npz_file=npz_file)


    data_split = [0.8, 0.1, 0.1]
    batch_size = 128
    use_pretrain = True

    here = Path(__file__).parent
    model_path = str(here / 'model' / 'nmr_model.pt')
    random_seed = 1
    if not os.path.exists(model_path): os.makedirs(model_path)

    data = GraphDataset()
    train_set, val_set, test_set = split_dataset(data, data_split, shuffle=True, random_state=random_seed)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    pred_loader = DataLoader(dataset=data_pred, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    train_y = np.hstack([inst[-2][inst[-1]] for inst in iter(train_loader.dataset)])
    train_y_mean = np.mean(train_y)
    train_y_std = np.std(train_y)

    node_dim = data.node_attr.shape[1]
    edge_dim = data.edge_attr.shape[1]
    net = nmrMPNN(node_dim, edge_dim).cuda()

    print('-- CONFIGURATIONS')
    print('--- data_size:', data.__len__())
    print('--- train/val/test: %d/%d/%d' %(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    print('--- use_pretrain:', use_pretrain)
    print('--- model_path:', model_path)


    # training
    if not use_pretrain:
        print('-- TRAINING')
        net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path)
    else:
        print('-- LOAD SAVED MODEL')
        net.load_state_dict(torch.load(model_path))


    # inference
    test_y = np.hstack([inst[-2][inst[-1]] for inst in iter(test_loader.dataset)])
    test_y_pred = inference(net, test_loader, train_y_mean, train_y_std)
    test_mae = mean_absolute_error(test_y, test_y_pred)

    print('-- RESULT')
    print('--- test MAE', test_mae)
    return inference(net,pred_loader, train_y_mean, train_y_std)

import pandas as pd
def main():
    parser = argparse.ArgumentParser(description='Convert a file containing SMILES strings into an SD file')
    parser.add_argument('-i','--input',dest='input_file', help='the input file. must be a csv containing the smiles')
    parser.add_argument('-o','--output',dest='output_file', help='writes the csv with the chemical_shift as output')
    args = parser.parse_args()
    input_csv = pd.read_csv(args.input_file)
    assert 'smiles' in ''.join(input_csv.columns).lower()
    smi_col = 'smiles' if 'smiles' in input_csv else 'canon_smiles' if 'canon_smiles' in input_csv else [col for col in input_csv.columns if 'smiles' in col.lower()][0]
    smiles = input_csv[smi_col].tolist()
    output_csv_fle = args.output_file
    chemical_shift = run_for_smiles(smiles)

    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    chemical_shift_col = []
    atm_i = 0
    for mol in mols:
        mol_row = []
        for _ in range(mol.GetNumAtoms()):
            mol_row.append(str(chemical_shift[atm_i]))
            atm_i += 1
        chemical_shift_col.append(",".join(mol_row))
    
    assert len(chemical_shift_col) == len(smiles)
    output_csv = input_csv.copy()
    output_csv["chemical_shift"] = chemical_shift_col
    output_csv.to_csv(output_csv_fle)
    

if __name__ == "__main__":
    main()

    #mols = ["CCCCOCCCC","c1ccccc1C=O",]
    #rslt = run_for_smiles(mols)
    #print(rslt)
    #print(len(rslt))
    #print([Chem.MolFromSmiles(mol).GetNumAtoms() for mol in mols])
