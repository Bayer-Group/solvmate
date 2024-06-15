import torch
import dgl
import numpy as np

                        
def collate_reaction_graphs(batch):

    gs, n_nodes, shifts, masks = map(list, zip(*batch))
    
    gs = dgl.batch(gs)

    n_nodes = torch.LongTensor(np.hstack(n_nodes))
    shifts = torch.FloatTensor(np.hstack(shifts))
    masks = torch.BoolTensor(np.hstack(masks))
    
    return gs, n_nodes, shifts, masks


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass