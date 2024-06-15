import argparse
import pandas as pd
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

from sklearn.metrics import mean_absolute_error,r2_score
from scipy import stats



from smal.all import random_fle
import numpy as np
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dgl.nn.pytorch import NNConv, Set2Set

from util import MC_dropout
from sklearn.metrics import mean_absolute_error,r2_score
from scipy import stats


import torch.nn as nn

from dgl.nn.pytorch import Set2Set

from dgllife.model.gnn.mpnn import MPNNGNN


class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )



    #def forward(self, g, node_feats, edge_feats):
    def forward(self, g, ):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = g.ndata['node_attr']
        edge_feats = g.edata['edge_attr']
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)

class nmrMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 node_feats = 64, embed_feats = 256,
                 num_step_message_passing = 5,
                 num_step_set2set = 3, num_layer_set2set = 1,
                 hidden_feats = 512, prob_dropout = 0.1):
        
        super(nmrMPNN, self).__init__()


        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, node_feats), nn.Tanh()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, embed_feats), nn.ReLU(),
            nn.Linear(embed_feats, node_feats * node_feats), nn.ReLU()
        )
        
        self.gnn_layer = NNConv(
            in_feats = node_feats,
            out_feats = node_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.gru = nn.GRU(node_feats, node_feats)
        
        self.readout = Set2Set(input_dim = node_feats * (1 + num_step_message_passing),
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)
                               
        self.predict = nn.Sequential(
            nn.Linear(node_feats * (1 + num_step_message_passing) * 3, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(hidden_feats, 1)
        )                           
                               
    def forward(self, g, n_nodes):
        
        def embed(g):
            
            node_feats = g.ndata['node_attr']
            node_feats = self.project_node_feats(node_feats)

            edge_feats = g.edata['edge_attr']

            node_aggr = [node_feats]
            for _ in range(self.num_step_message_passing):
                msg = self.gnn_layer(g, node_feats, edge_feats).unsqueeze(0)
                _, node_feats = self.gru(msg, node_feats.unsqueeze(0))
                node_feats = node_feats.squeeze(0)
                
                node_aggr.append(node_feats)
                
            node_aggr = torch.cat(node_aggr, 1)
            
            return node_aggr

        node_embed_feats = embed(g)
        graph_embed_feats = self.readout(g, node_embed_feats)        
        graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)

        out = self.predict(torch.hstack([node_embed_feats, graph_embed_feats])).flatten()

        return out

        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, n_forward_pass = 5, cuda = torch.device('cuda:0')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    optimizer = Adam(net.parameters(), lr=1e-5, weight_decay=1e-10)
    print("applying decreasing lr scheme across layers...")
    optimizer = Adam(
        [
        {"params": net.project_node_feats.parameters(), "lr":1e-6},
        {"params": net.gnn_layer.parameters(), "lr":1e-5},
        {"params": net.gru.parameters(), "lr":1e-4},
        {"params": net.readout.parameters(), "lr":1e-4},
        {"params": net.predict.parameters(), "lr":1e-3},
        ],
        lr=1e-5,
        weight_decay=1e-10,
    )
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6, verbose=True)

    max_epochs = 500
    val_y = np.hstack([inst[-2][inst[-1]] for inst in iter(val_loader.dataset)])
    val_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs, n_nodes, y = batchdata
            
            y = (y - train_y_mean) / train_y_std
            
            inputs = inputs.to(cuda)
            n_nodes = n_nodes.to(cuda)
            y = y.to(cuda)
            
            predictions = net(inputs,) # n_nodes, y)
            
            loss = torch.abs(predictions - y).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item() * train_y_std

        #print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  train_size, train_size, train_loss, (time.time()-start_time)/60))
    
        # validation
        val_y_pred = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
        val_loss = mean_absolute_error(val_y, val_y_pred)
        val_r2 = r2_score(val_y, val_y_pred)
        val_spearmanr = stats.spearmanr(val_y, val_y_pred)[0]
        
        val_log[epoch] = val_loss
        if epoch % 10 == 0: 
            print('--- validation epoch %d, processed %d, current MAE %.3f, current r2 %.3f, current spearr %.3f, best MAE %.3f, time elapsed(min) %.2f' %(epoch, val_loader.dataset.__len__(), val_loss, val_r2, val_spearmanr, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
        
        lr_scheduler.step(val_loss)
        
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            break

    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    net.eval()
    MC_dropout(net)
    tsty_pred = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs = batchdata[0].to(cuda)
            n_nodes = batchdata[1].to(cuda)

            tsty_pred.append(np.array([net(inputs, n_nodes).cpu().numpy() for _ in range(n_forward_pass)]).transpose())

    tsty_pred = np.vstack(tsty_pred) * train_y_std + train_y_mean
    
    return np.mean(tsty_pred, 1)




def run_for_smiles(smis:list[str],):
    data_split = [0.8, 0.1, 0.1]
    batch_size = 128
    use_pretrain = True

    here = Path(__file__).parent
    model_path = str(here / 'checkpoints' / 'model.pt')
    random_seed = 1
    #if not os.path.exists(model_path): os.makedirs(model_path)
    data_pred = pd.DataFrame({"solute SMILES": smis, "conc": [0 for _ in smis]})
    data_pred = GraphDataset(data_pred)
    data = pd.read_csv(here /  "data" / "training_data_singleton.csv")
    data = GraphDataset(data)
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
    net = MPNNPredictor(node_dim, edge_dim).cuda()

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


if __name__ == "__main__":
    run_for_smiles(["CCOCC"])