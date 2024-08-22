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


data_split = [0.8, 0.1, 0.1]
batch_size = 128
use_pretrain = False
model_path = './model/nmr_model.pt'
random_seed = 1
if not os.path.exists('./model/'): os.makedirs('./model/')

data = GraphDataset()
train_set, val_set, test_set = split_dataset(data, data_split, shuffle=True, random_state=random_seed)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

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
if use_pretrain == False:
    print('-- TRAINING')
    print("using pretrained NMR chemical shift model:")
    net.load_state_dict(torch.load('./model/nmr_model__original.pt'))
    net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path)
else:
    print('-- LOAD SAVED MODEL')
    net.load_state_dict(torch.load(model_path))


# inference
test_y = np.hstack([inst[-2][inst[-1]] for inst in iter(test_loader.dataset)])
test_y_pred = inference(net, test_loader, train_y_mean, train_y_std)
test_mae = mean_absolute_error(test_y, test_y_pred)
test_r2 = r2_score(test_y, test_y_pred)
test_spearmanr = stats.spearmanr(test_y, test_y_pred)[0]


print('-- RESULT')
print('--- test MAE', test_mae)
print('--- test R2', test_r2)
print('--- test spearmanr', test_spearmanr)