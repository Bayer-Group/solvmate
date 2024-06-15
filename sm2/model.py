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
                               
    def forward(self, g, n_nodes, masks):
        
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

        out = self.predict(torch.hstack([node_embed_feats, graph_embed_feats])[masks]).flatten()

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

            inputs, n_nodes, shifts, masks = batchdata
            
            shifts = (shifts[masks] - train_y_mean) / train_y_std
            
            inputs = inputs.to(cuda)
            n_nodes = n_nodes.to(cuda)
            shifts = shifts.to(cuda)
            masks = masks.to(cuda)
            
            predictions = net(inputs, n_nodes, masks)
            
            loss = torch.abs(predictions - shifts).mean()
            
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
            masks = batchdata[3].to(cuda)

            tsty_pred.append(np.array([net(inputs, n_nodes, masks).cpu().numpy() for _ in range(n_forward_pass)]).transpose())

    tsty_pred = np.vstack(tsty_pred) * train_y_std + train_y_mean
    
    return np.mean(tsty_pred, 1)
