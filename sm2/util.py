from pathlib import Path
import joblib
import torch
import dgl
import numpy as np

                        
def collate_reaction_graphs(batch):

    gs, n_nodes, shifts = map(list, zip(*batch))
    
    gs = dgl.batch(gs)

    n_nodes = torch.LongTensor(np.hstack(n_nodes))
    shifts = torch.FloatTensor(np.hstack(shifts))
    
    return gs, n_nodes, shifts


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
