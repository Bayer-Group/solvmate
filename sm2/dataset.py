import os, sys
from smal.all import from_smi
import numpy as np
import pandas as pd
import torch
from dgl.convert import graph
from pathlib import Path
import pandas as pd
from rdkit import Chem
import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem, ChemicalFeatures
import numpy as np
import os
import torch


import numpy as np

import torch


class GraphDataset():

    def __init__(self, df:pd.DataFrame,):
        self._read_dataframe(df)
        self._load()


    def _read_dataframe(self, df:pd.DataFrame):
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

            #shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
            #mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

            mol_dict['n_node'].append(n_node)
            mol_dict['n_edge'].append(n_edge)
            mol_dict['node_attr'].append(node_attr)

            #mol_dict['shift'].append(shift)
            #mol_dict['mask'].append(mask)
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

            else:
                assert False
            
            return mol_dict


        df["mol"] = df["solute SMILES"].apply(from_smi)
        molsuppl = df["mol"].tolist()

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
                    #'shift': [],
                    #'mask': [],
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
                #assert '.' not in Chem.MolToSmiles(mol) # TODO
            except:
                raise

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
        #mol_dict['shift'] = np.hstack(mol_dict['shift'])
        #mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
        mol_dict['smi'] = np.array(mol_dict['smi'])
        mol_dict["conc"] = np.array(df["conc"].tolist())

        for key in mol_dict.keys(): 
            print(key, mol_dict[key].shape, mol_dict[key].dtype)
        
        self.mol_dict = mol_dict
            




    def _load(self):
        mol_dict = self.mol_dict
        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
                
        self.conc = mol_dict["conc"]
        #self.shift = mol_dict['shift']
        #self.mask = mol_dict['mask']
        self.smi = mol_dict['smi']

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
        

    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx:idx+1].astype(int)
        #shift = self.shift[self.n_csum[idx]:self.n_csum[idx+1]].astype(float)
        #mask = self.mask[self.n_csum[idx]:self.n_csum[idx+1]]
        
        return g, n_node, self.conc[idx] #, shift, mask
        
        
    def __len__(self):

        return self.n_node.shape[0]

class SMDataset():

    def __init__(self, df:pd.DataFrame,):
        self._read_dataframe(df)


    def _read_dataframe(self, df:pd.DataFrame):
        def _read_mol_dict(molsuppl):
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

                #shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
                #mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

                mol_dict['n_node'].append(n_node)
                mol_dict['n_edge'].append(n_edge)
                mol_dict['node_attr'].append(node_attr)

                #mol_dict['shift'].append(shift)
                #mol_dict['mask'].append(mask)
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
                        #'shift': [],
                        #'mask': [],
                        'smi': []}
                            
            for i, mol in enumerate(molsuppl):
                if isinstance(mol,str):
                    mol = from_smi(mol)

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
                    #assert '.' not in Chem.MolToSmiles(mol) # TODO
                except:
                    raise

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
            #mol_dict['shift'] = np.hstack(mol_dict['shift'])
            #mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
            mol_dict['smi'] = np.array(mol_dict['smi'])

            mol_dict['n_csum'] = np.concatenate([[0], np.cumsum(mol_dict['n_node'])])
            mol_dict['e_csum'] = np.concatenate([[0], np.cumsum(mol_dict['n_edge'])])

            for key in mol_dict.keys(): 
                print(key, mol_dict[key].shape, mol_dict[key].dtype)
            
            return mol_dict

        self.mol_dict_solva = _read_mol_dict(df["solvent SMILES a"])
        self.mol_dict_solvb = _read_mol_dict(df["solvent SMILES b"])
        self.mol_dict_solu = _read_mol_dict(df["solute SMILES"])
        self.conc = df["conc"].values

    def _load_graph(self,mol_dict:dict,idx:int,):
        e_csum = mol_dict["e_csum"]
        n_csum = mol_dict["n_csum"]
        n_node = mol_dict["n_node"]
        src = mol_dict["src"]
        dst = mol_dict["dst"]
        node_attr = mol_dict["node_attr"]
        edge_attr = mol_dict["edge_attr"]
        g = graph((src[e_csum[idx]:e_csum[idx+1]], dst[e_csum[idx]:e_csum[idx+1]]), num_nodes = n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(node_attr[n_csum[idx]:n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(edge_attr[e_csum[idx]:e_csum[idx+1]]).float()
        return g

    def __getitem__(self, idx):
        conc = self.conc[idx]
        g_solva = self._load_graph(self.mol_dict_solva,idx)
        g_solvb = self._load_graph(self.mol_dict_solvb,idx)
        g_solu = self._load_graph(self.mol_dict_solu,idx)
        return g_solu, g_solva, g_solvb, conc
        
        
    def __len__(self):

        return len(self.conc)