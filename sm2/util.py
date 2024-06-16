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