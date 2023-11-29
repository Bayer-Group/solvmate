import random
import copy
import pathlib
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F

from solvmate.ccryst import datasets,info,chem_utils

FIG_PATH = pathlib.Path.home() / "figures"


def load_data(top_n=20,downsample_mantas=0,):
    """
    loads the data and applies basic downsampling and top_n
    solvent removal if requested
    """
    info.log("loading data",elt="header",)
    
    df = datasets.load_all(verbose=False,)
    info.log("loaded data",elt="header",)
    info.log("Source counts in beginning:")
    info.log(df.source.value_counts())
    info.log()
      
    if downsample_mantas:
        df = datasets.downsample_sources(df,["mantas"],downsample_mantas,)
        info.log("Source counts after downsampling mantas:")
        info.log(df.source.value_counts())
        info.log()
    
    
    datasets.add_split_by_col(df)
    info.log("Added train test split:")
    info.log(df.split.value_counts())
    
    if top_n:
        top_solvents = df[~df.source.isin(["mantas","cod"])].solvent_label.value_counts().index[0:top_n].tolist()
        info.log(f"Restricted to the top_n={top_n} solvents: {top_solvents}")
        info.log(f"#rows_before = {len(df)}")
        df = df[df.solvent_label.isin(top_solvents)]
        info.log(f"#rows_after = {len(df)}")

    df["ecfp"] = df.mol_compound.apply(chem_utils.ecfp_fingerprint)
        
    return df

class XYDataSet(data.Dataset):
    
    def __init__(self,dat):
        """
        Assumes a dataframe to be passed
        where X is the input col and
        Y is the target col.
        """
        
        super().__init__()
        self.dat = dat
        
    def __len__(self):
        return len(self.dat)
    
    def __getitem__(self,idx):
        dat = self.dat
        x = dat["X"].iloc[idx]
        y = dat["Y"].iloc[idx]
        return torch.Tensor(x),y



class NetEmber(nn.Module):
    
    def __init__(self, dim_in, dim_emb, dim_out):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_emb = dim_emb
        
        self.fc_in = nn.Linear(dim_in,dim_emb,)
        self.fc_e1 = nn.Linear(dim_emb,dim_emb,)
        self.fc_out = nn.Linear(dim_emb,dim_out)
        
    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(x)
        
        x = self.fc_e1(x)
        x = F.relu(x)
        
        x = self.fc_out(x)
        return x


def run_epoch(net, criterion, optimizer, n_epoch, dl, phase, dataset_sizes,scheduler=None):
    assert phase in ["train","test"]

    if phase == 'train':
        net.train()  # Set model to training mode
    else:
        net.eval()
    
    running_loss = 0.0
    running_corrects = 0
    for X,y in dl:
        optimizer.zero_grad()
        
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = net(X)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)
        if phase == 'train' and scheduler:
            scheduler.step()

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    info.log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # deep copy the model
    if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        
def train(net, criterion, optimizer, data_loaders, dataset_sizes,n_epochs,scheduler=None):
    for n_epoch in range(n_epochs):
        for phase in ["train","test"]:
            run_epoch(net, criterion, optimizer, n_epoch+1, data_loaders[phase], phase, dataset_sizes,scheduler=scheduler)


def main():
    info.LOG_LEVEL = info.DEBUG
    BATCH_SIZE = 64
    NUM_WORKERS = 1
    TOP_N = 20
    N_EPOCHS = 10


    df = load_data(top_n=TOP_N,downsample_mantas=10000,)
    df["X"] = df.ecfp
    df["Y"] = df.bin_cryst

    clf_dict = {}
    for solvent in df.solvent_label.unique():
        
        df_solvent = df[df.solvent_label == solvent]
        
        info.log(f"solvent = {solvent}",elt="header")
        info.log(f"bin_cryst.value_counts = {df_solvent.bin_cryst.value_counts()}",)
        
        ds_train = XYDataSet(df_solvent[df_solvent.split == "train"])
        ds_test = XYDataSet(df_solvent[df_solvent.split == "test"])
        
        dl_train = data.DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,)
        dl_test = data.DataLoader(ds_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,)
        

        net = NetEmber(dim_in=2048,dim_emb=512,dim_out=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())
        scheduler = None
        dataset_sizes = {"train": len(ds_train), "test": len(ds_test)}
        data_loaders = {"train": dl_train, "test": dl_test}

        train(net=net, criterion=criterion, optimizer=optimizer, data_loaders=data_loaders,
            dataset_sizes=dataset_sizes,n_epochs=N_EPOCHS,scheduler=scheduler)

if __name__ == "__main__":
    main()