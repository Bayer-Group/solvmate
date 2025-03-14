import argparse
import joblib
import pandas as pd
from pathlib import Path
from dataset import GraphDataset, SMDataset
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
from abs_model import run_for_smiles_abs
from util import collate_reaction_graphs, store_result, path_for_experiment_and_fle

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

class SMPredictor(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(SMPredictor, self).__init__()

        self.gnn_solu = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)

        self.gnn_solv_a = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)

        self.gnn_solv_b = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)

        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            # 2 (=? ) * 3 (= solv_a,solv_b,solu) + 2 (= temp_a,temp_b)
            nn.Linear(2 * 3 * node_out_feats + 2, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )

    def forward(self, g_solu, g_solv_a1, g_solv_a2, g_solv_a_facs, g_solv_b1, g_solv_b2, g_solv_b_facs, temp_a, temp_b,):
        node_feats_solu = g_solu.ndata['node_attr']
        edge_feats_solu = g_solu.edata['edge_attr']
        node_feats_solu = self.gnn_solu(g_solu, node_feats_solu, edge_feats_solu)
        graph_feats_solu = self.readout(g_solu, node_feats_solu)

        # Solvent Block A
        node_feats_solv_a1 = g_solv_a1.ndata['node_attr']
        edge_feats_solv_a1 = g_solv_a1.edata['edge_attr']
        node_feats_solv_a1 = self.gnn_solv_a(g_solv_a1, node_feats_solv_a1, edge_feats_solv_a1)
        graph_feats_solv_a1 = self.readout(g_solv_a1, node_feats_solv_a1)

        node_feats_solv_a2 = g_solv_a2.ndata['node_attr']
        edge_feats_solv_a2 = g_solv_a2.edata['edge_attr']
        node_feats_solv_a2 = self.gnn_solv_a(g_solv_a2, node_feats_solv_a2, edge_feats_solv_a2)
        graph_feats_solv_a2 = self.readout(g_solv_a2, node_feats_solv_a2)

        graph_feats_solv_a = g_solv_a_facs[:,0].reshape(-1,1) * graph_feats_solv_a1 + g_solv_a_facs[:,1].reshape(-1,1) * graph_feats_solv_a2

        # Solvent Block B
        node_feats_solv_b1 = g_solv_b1.ndata['node_attr']
        edge_feats_solv_b1 = g_solv_b1.edata['edge_attr']
        node_feats_solv_b1 = self.gnn_solv_b(g_solv_b1, node_feats_solv_b1, edge_feats_solv_b1)
        graph_feats_solv_b1 = self.readout(g_solv_b1, node_feats_solv_b1)

        node_feats_solv_b2 = g_solv_b2.ndata['node_attr']
        edge_feats_solv_b2 = g_solv_b2.edata['edge_attr']
        node_feats_solv_b2 = self.gnn_solv_b(g_solv_b2, node_feats_solv_b2, edge_feats_solv_b2)
        graph_feats_solv_b2 = self.readout(g_solv_b2, node_feats_solv_b2)

        graph_feats_solv_b = g_solv_b_facs[:,0].reshape(-1,1) * graph_feats_solv_b1 + g_solv_b_facs[:,1].reshape(-1,1) * graph_feats_solv_b2


        # A simple StdScaler that assumes mean temperatures of 30 degrees
        # and standard deviations of 15 degrees. Should project most
        # reasonable temperatures into a reasonable range
        temp_a = (temp_a - 30) / 15
        temp_b = (temp_b - 30) / 15

        assert len(temp_a) == len(temp_b)
        assert len(temp_a) == len(graph_feats_solu)
        return self.predict(torch.hstack([graph_feats_solu,graph_feats_solv_a,graph_feats_solv_b,temp_a.reshape(-1,1),temp_b.reshape(-1,1),]))


        
def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, n_forward_pass = 5, cuda = torch.device('cuda:0'), experiment_name=None):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    optimizer = Adam(net.parameters(), lr=1e-3,)# weight_decay=1e-10)

    max_epochs = 500
    val_y = np.hstack([inst[-1] for inst in iter(val_loader.dataset)])
    val_log = np.zeros(max_epochs)
    train_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            optimizer.zero_grad()
            g_solu, g_solva_1, g_solva_2, mixture_coefficients_a, g_solvb_1, g_solvb_2, mixture_coefficients_b, temp_a, temp_b, y = batchdata
            
            y = (y - train_y_mean) / train_y_std
            
            g_solu = g_solu.to(cuda)
            g_solva_1 = g_solva_1.to(cuda)
            g_solva_2 = g_solva_2.to(cuda)
            g_solvb_1 = g_solvb_1.to(cuda)
            g_solvb_2 = g_solvb_2.to(cuda)

            mixture_coefficients_a = mixture_coefficients_a.to(cuda)
            mixture_coefficients_b = mixture_coefficients_b.to(cuda)
            temp_a = temp_a.to(cuda)
            temp_b = temp_b.to(cuda)
            y = y.to(cuda)
            
            predictions = net(g_solu,g_solva_1,g_solva_2,mixture_coefficients_a,g_solvb_1,g_solvb_2,mixture_coefficients_b,temp_a,temp_b,) 
            
            loss = torch.abs(predictions.squeeze() - y).mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item() * train_y_std
            train_log[epoch] = train_loss


        #print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  train_size, train_size, train_loss, (time.time()-start_time)/60))
    
        # validation
        val_y_pred = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
        val_loss = mean_absolute_error(val_y, val_y_pred)
        val_r2 = r2_score(val_y, val_y_pred)
        val_spearmanr = stats.spearmanr(val_y, val_y_pred)[0]
        
        val_log[epoch] = val_loss
        if epoch % 10 == 0: 
            print('--- validation epoch %d, processed %d, current MAE %.3f, current r2 %.3f, current spearr %.3f, best MAE %.3f, time elapsed(min) %.2f' %(epoch, val_loader.dataset.__len__(), val_loss, val_r2, val_spearmanr, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
        
        #lr_scheduler.step(val_loss)
        
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            break

    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))

    from matplotlib import pyplot as plt
    plt.clf()
    plt.plot(val_log,"b-o")
    plt.plot(train_log,"r-o")
    plt.savefig(path_for_experiment_and_fle(experiment_name=experiment_name,fle_name="loss_curve.svg"))
    plt.clf()
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    if not torch.cuda.is_available():
        cuda = torch.device("cpu")

    net.eval()
    #MC_dropout(net)
    with torch.no_grad():
        y_pred = []
        for batchidx, batchdata in enumerate(test_loader):

            g_solu, g_solva1, g_solva2, fac_a, g_solvb1 ,g_solvb2, fac_b, temp_a, temp_b, _ = batchdata
            g_solu = g_solu.to(cuda)
            g_solva1 = g_solva1.to(cuda)
            g_solva2 = g_solva2.to(cuda)
            fac_a = fac_a.to(cuda)
            g_solvb1 = g_solvb1.to(cuda)  
            g_solvb2 = g_solvb2.to(cuda)  
            fac_b = fac_b.to(cuda)
            temp_a = temp_a.to(cuda)
            temp_b = temp_b.to(cuda)
            
            predictions = net(g_solu,g_solva1,g_solva2,fac_a,g_solvb1,g_solvb2,fac_b,temp_a,temp_b,) # n_nodes, y)
            y_pred.append(predictions.cpu().numpy())

    y_pred_inv_std = np.vstack(y_pred) * train_y_std + train_y_mean
    return y_pred_inv_std
    



def evaluate_ranking(df:pd.DataFrame,experiment_name:str,):
    import seaborn as sns
    from matplotlib import pyplot as plt
    rs = []
    for smi in df["solute SMILES"].unique():
        g = df[df["solute SMILES"] == smi]
        r = stats.spearmanr(g["conc"], g["conc_pred"])[0]
        rs.append(r)

    plt.clf()
    sns.violinplot(rs)
    plt.savefig(path_for_experiment_and_fle(experiment_name=experiment_name,fle_name="ranking_spearmanrs_onb_only.svg"))
    plt.clf()






from smal.all import add_split_by_col
def run_for_smiles(smis:list[str],experiment_name:str,):
    batch_size = 128
    use_pretrain = True

    here = Path(__file__).parent
    model_path = str(here / 'checkpoints' / 'model.pt')
    model_metadata_path = str(here / 'checkpoints' / 'model_metadata.pkl')
    #if not os.path.exists(model_path): os.makedirs(model_path)
    data_pred = pd.DataFrame({"solute SMILES": smis, "temp_a": 25, "temp_b": 25,})
    data_pred["solvent SMILES a"] = "CO.CCO" # TODO
    data_pred["solvent SMILES b"] = "CCCCCO.CCO" # TODO
    data_pred["conc"] = 0
    data_pred["mixture_coefficients a"] = [[1] for _ in data_pred.iterrows()]
    data_pred["mixture_coefficients b"] = [[1] for _ in data_pred.iterrows()]
    data_pred = SMDataset(data_pred)
    #data_pred = pd.DataFrame({"solute SMILES": smis, "solvent SMILES": ["CCO" for _ in smis], "conc": [0 for _ in smis]})
    #data_pred = GraphDataset(data_pred)
    #data = pd.read_csv(here /  "data" / "training_data_singleton.csv")
    #data = GraphDataset(data)
    data = pd.read_csv(here /  "data" / "sm2_pairwise.csv")

    #data = data.sample(1000,random_state=123)

    #data = data[data["source"] == "open_notebook"]
    # data = data[data["source"] == "nova"]
    assert len(data)

    smiles_blacklist = ["[Na]Cl",]
    for col in ["solute SMILES", "solvent SMILES a", "solvent SMILES b",]:
        data = data[~data[col].isin(smiles_blacklist)]

    for col in ['mixture_coefficients a', 'mixture_coefficients b',]:
        data[col] = data[col].apply(eval)

    data["conc"] = data["conc diff"]
    add_split_by_col(data,col="solute SMILES",amount_train=0.6,amount_test=0.2,amount_val=0.2,random_seed=123,)

    df_test = data[data["split"] == "test"]
    df_train = data[data["split"] == "train"]
    df_val = data[data["split"] == "val"]

    train_set = SMDataset(df_train)
    val_set = SMDataset(df_val)
    test_set = SMDataset(df_test)
    data = SMDataset(data)

    assert len(train_set)
    assert len(val_set)
    assert len(test_set)
    
    #data = SMDataset(data)
    #train_set, val_set, test_set = split_dataset(data, data_split, shuffle=True, random_state=random_seed)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    pred_loader = DataLoader(dataset=data_pred, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    train_y = np.hstack([inst[-1] for inst in iter(train_loader.dataset)])
    train_y_mean = np.mean(train_y.reshape(-1))
    train_y_std = np.std(train_y.reshape(-1))

    node_dim = data.mol_dict_solu["node_attr"].shape[1]
    edge_dim = data.mol_dict_solu["edge_attr"].shape[1]
    net = SMPredictor(node_dim, edge_dim).cuda()

    print('-- CONFIGURATIONS')
    print('--- data_size:', data.__len__())
    print('--- train/val/test: %d/%d/%d' %(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    print('--- use_pretrain:', use_pretrain)
    print('--- model_path:', model_path)

    joblib.dump({"train_y_mean": train_y_mean, "train_y_std": train_y_std}, model_metadata_path,)

    # training
    if not use_pretrain or not Path(model_path).exists():
        print('-- TRAINING')
        net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path,experiment_name=experiment_name,)
    else:
        print('-- LOAD SAVED MODEL')
        net.load_state_dict(torch.load(model_path))



    # inference
    test_y = np.hstack([inst[-1] for inst in iter(test_loader.dataset)])
    test_y_pred = inference(net, test_loader, train_y_mean, train_y_std)
    test_mae = mean_absolute_error(test_y, test_y_pred)

    df_test["conc_pred"] = test_y_pred

    evaluate_ranking(df_test,experiment_name=experiment_name,)

    print('-- RESULT')
    print('--- test MAE', test_mae)
    return inference(net,pred_loader, train_y_mean, train_y_std)



def _least_squares_solution(g_solu:pd.DataFrame,id_col:str,include_abs_calcs=True,):
    """
    Orders the given per-solute dataframe by applying least squares
    procedure on the pairwise differences.

    >>> order = "abcd"
    >>> g = pd.DataFrame([{"solvent SMILES a": val_a, "solvent SMILES b": val_b, "conc": order.index(val_b) - order.index(val_a)} for val_a in order for val_b in order])
    >>> _least_squares_solution(g,id_col="solvent SMILES",include_abs_calcs=False,) #["conc"].tolist() # doctest: +NORMALIZE_WHITESPACE
        solvent SMILES  conc
    3              d   1.5
    2              c   0.5
    1              b  -0.5
    0              a  -1.5
    
    """
    id_col_a = " ".join([id_col,"a"])
    id_col_b = " ".join([id_col,"b"])
    assert id_col_a in g_solu.columns
    assert id_col_b in g_solu.columns
    g_solu_us = g_solu.drop_duplicates(id_col_b)
    solv_to_idx = g_solu_us[id_col_b].tolist()
    solv_to_idx = {smi: idx for idx,smi in enumerate(solv_to_idx)}
    N = len(solv_to_idx)
    M = []
    b = []

    for _,row in g_solu.iterrows():
        m = np.zeros(N)
        ia = solv_to_idx[row[id_col_a]]
        ib = solv_to_idx[row[id_col_b]]
        m[ia] = - 1
        m[ib] = 1
        
        b.append(row["conc"])
        M.append(m)
    
    N_exps = g_solu[id_col_b].nunique()
    assert N_exps == N
    if include_abs_calcs:
        # If absolute calculations are to be included when solving the lin eqs,
        # then this means that in addition to the pairwise terms a -b  = dS
        # we also add terms of the a 0 = S
        # for each unique solvent smiles that we found
        M_abs = np.eye(N)
        b_abs = run_for_smiles_abs(
            smis=g_solu_us["solute SMILES"].tolist(),
            experiment_name="preds_",
            solvent_smis=g_solu_us["solvent SMILES b"].tolist(),
            facs=g_solu_us["mixture coefficients b"].tolist(),
            temps=g_solu_us["temp b"].tolist(),
            )
        for m in M_abs:
            M.append(m)
        for row in b_abs:
            b.append(float(row))

    M = np.vstack(M)
    solution = np.linalg.lstsq(M, b, rcond=None)[0]
    #g_solu["absolute_ordering"] = g_solu["solvent SMILES b"].apply(lambda smi: solution[solv_to_idx[smi]])

    return pd.DataFrame([{
        id_col: smi,
        "conc": solution[idx],
    }
        for smi,idx in solv_to_idx.items()
    ]
    ).sort_values("conc",ascending=False,)


def run_predictions_for_solvents(solute_smiles:str, solvents:list[str], temps:list[float]=None,facs:list[list[float]]=None,):
    solvent_smiles_a = [solv_a for solv_a in solvents for solv_b in solvents ]
    solvent_smiles_b = [solv_b for solv_a in solvents for solv_b in solvents ]

    if temps is None:
        temps = [0 for _ in solvents]

    if facs is None:
        facs = [[1] for _ in solvents]

    temps_a = [temps[i] for i,_ in enumerate(solvents) for _ in solvents]
    temps_b = [temps[i] for _ in solvents for i,_ in enumerate(solvents)]

    facs_a = [facs[i] for i,_ in enumerate(solvents) for _ in solvents]
    facs_b = [facs[i] for _ in solvents for i,_ in enumerate(solvents)]

    assert isinstance(solute_smiles,str)
    solute_smiles = [solute_smiles for _ in solvent_smiles_a]

    log_s = run_predictions_for_smiles_pairs(solute_smiles=solute_smiles,
                                             solvent_smiles_a=solvent_smiles_a,
                                             solvent_smiles_b=solvent_smiles_b,
                                             facs_a=facs_a, facs_b=facs_b,
                                             temps_a=temps_a,temps_b=temps_b,
                                             )
    dfp = pd.DataFrame({
        "solvent SMILES a": solvent_smiles_a,
        "solvent SMILES b": solvent_smiles_b,
        "temp a": temps_a,
        "temp b": temps_b,
        "log S": list(log_s.reshape(-1)),
        "solute SMILES": solute_smiles,
        "mixture coefficients a": facs_a,
        "mixture coefficients b": facs_b,
    })

    dfp["exp_id a"] = dfp["solvent SMILES a"] + "__" + dfp["temp a"].apply(str)
    dfp["exp_id b"] = dfp["solvent SMILES b"] + "__" + dfp["temp b"].apply(str)

    dfo = []
    for solu in dfp["solute SMILES"].unique():
        g_solu = dfp[dfp["solute SMILES"] == solu]
        g_solu["conc"] = g_solu["log S"]
        g_rank = _least_squares_solution(g_solu, "exp_id")
        g_rank["log S"] = g_rank["conc"]
        g_rank["solute SMILES"] = solu
        dfo.append(g_rank)
        #for solv in dfp["solvent SMILES b"].unique():
        # g_solu_solv = g_solu[g_solu["solvent SMILES b"] == solv]
        # dfo.append({"solute SMILES": solu, "solvent SMILES": solv, "log S": g_solu_solv["log S"].sum(),})
    
    dfo = pd.concat(dfo)
    dfo["solvent SMILES"] = dfo["exp_id"].apply(lambda ei: ei.split("__")[0])
    dfo["temp"] = dfo["exp_id"].apply(lambda ei: ei.split("__")[1])
    return dfo
        


def run_predictions_for_smiles_pairs(solute_smiles:list[str], solvent_smiles_a:list[str], solvent_smiles_b:list[str], facs_a:list[list[float]], facs_b:list[list[float]], temps_a:list[float], temps_b:list[float], ):
    batch_size = 128
    here = Path(__file__).parent
    model_path = str(here / 'checkpoints' / 'model.pt')
    model_metadata_path = str(here / 'checkpoints' / 'model_metadata.pkl')
    data_pred = pd.DataFrame(
        {
            "solute SMILES": solute_smiles,
            "solvent SMILES a": solvent_smiles_a,
            "solvent SMILES b": solvent_smiles_b,
            "temp_a": temps_a,
            "temp_b": temps_b,
            "mixture_coefficients a": facs_a,
            "mixture_coefficients b": facs_b,
        }
        )
    data_pred["conc"] = 0
    data_pred = SMDataset(data_pred)

    pred_loader = DataLoader(dataset=data_pred, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    node_dim = data_pred.mol_dict_solu["node_attr"].shape[1]
    edge_dim = data_pred.mol_dict_solu["edge_attr"].shape[1]
    if torch.cuda.is_available():
        net = SMPredictor(node_dim, edge_dim).cuda()
        net.load_state_dict(torch.load(model_path),)
    else:
        net = SMPredictor(node_dim, edge_dim)
        net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    metadata = joblib.load(model_metadata_path,)
    train_y_mean = metadata["train_y_mean"]
    train_y_std = metadata["train_y_std"]
    return inference(net,pred_loader, train_y_mean, train_y_std)


if __name__ == "__main__":

    print(
        run_for_smiles(["CCOCC"],experiment_name="differential_model_onb_only",)
    )