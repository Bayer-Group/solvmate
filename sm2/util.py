from pathlib import Path
import joblib
import torch
import dgl
import numpy as np

                        
def collate_reaction_graphs(batch):
    g_solu, g_solva_1, g_solva_2, facs_a, g_solvb_1, g_solvb_2, facs_b, conc = map(list, zip(*batch))
    
    g_solu = dgl.batch(g_solu)
    g_solva_1 = dgl.batch(g_solva_1)
    g_solvb_1 = dgl.batch(g_solvb_1)
    g_solva_2 = dgl.batch(g_solva_2)
    g_solvb_2 = dgl.batch(g_solvb_2)

    conc = torch.FloatTensor(np.hstack(conc))
    facs_a = torch.FloatTensor(np.array(facs_a))
    facs_b = torch.FloatTensor(np.array(facs_b))
    return g_solu, g_solva_1, g_solva_2, facs_a, g_solvb_1, g_solvb_2, facs_b, conc


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass

def path_for_experiment_and_fle(experiment_name:str, fle_name:str):
    here = Path(__file__).parent
    path_results = Path(here) / "results"
    path_exp =  path_results / experiment_name 
    path_exp.mkdir(exist_ok=True,)
    path_obj = path_exp / fle_name
    return path_obj

def store_result(obj:"any", experiment_name:str, fle_name:str) -> None:
    path_obj = path_for_experiment_and_fle(experiment_name=experiment_name,fle_name=fle_name,)
    joblib.dump(obj,path_obj)

def load_result(experiment_name:str, fle_name:str) -> "any":
    path_obj = path_for_experiment_and_fle(experiment_name=experiment_name,fle_name=fle_name,)
    return joblib.load(path_obj)
