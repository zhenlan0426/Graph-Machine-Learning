#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 07:11:38 2022

@author: will
"""

from torch_geometric.datasets import IMDB
from torch_geometric.nn import GATv2Conv
from torch_sparse import SparseTensor
import torch
from torch.nn import Embedding,Sequential, Linear, BatchNorm1d,Dropout,LeakyReLU,CrossEntropyLoss,Identity
from torch import nn
from torch.optim import Adam, SGD
import copy
from sklearn.metrics import f1_score
import numpy as np

# =============================================================================
# Data
# =============================================================================
def get_data(model,data):
    # data is HeteroData
    if isinstance(model,GNN1):
        homo_data = data.to_homogeneous()
        out_data = [homo_data.x,homo_data.edge_index,homo_data.node_type,homo_data.edge_type]
        return [d.to('cuda') for d in out_data]
    
    elif isinstance(model,GNN2):
        M,A,D = data['movie'].x, data['actor'].x, data['director'].x
        m2d_idx = data['movie', 'to', 'director']['edge_index']
        m2a_idx = data['movie', 'to', 'actor']['edge_index']
        d2m_idx = data['director', 'to', 'movie']['edge_index']
        a2m_idx = data['actor', 'to', 'movie']['edge_index']
        
        adj_m2d = SparseTensor(row=m2d_idx[1],col=m2d_idx[0])
        adj_m2a = SparseTensor(row=m2a_idx[1],col=m2a_idx[0])
        adj_d2m = SparseTensor(row=d2m_idx[1],col=d2m_idx[0])
        adj_a2m = SparseTensor(row=a2m_idx[1],col=a2m_idx[0])
        return [d.to('cuda') for d in [M,A,D,adj_m2d,adj_m2a,adj_d2m,adj_a2m]]
        

def MLP(in_d,out_d,multiple_factor,dropout=0):
    return Sequential(  Dropout(dropout) if dropout>0 else Identity(),
                        BatchNorm1d(in_d,track_running_stats=False),
                        Linear(in_d,in_d*multiple_factor),
                        LeakyReLU(inplace=True),
                        Dropout(dropout) if dropout>0 else Identity(),
                        BatchNorm1d(in_d*multiple_factor,track_running_stats=False),
                        Linear(in_d*multiple_factor,out_d),
                        LeakyReLU(inplace=True))

# =============================================================================
#  GNN1
# =============================================================================
class GAT_block(torch.nn.Module):
    def __init__(self,d,d_type,heads,dropout,multiple_factor=2):
        super(GAT_block, self).__init__()   
        self.v_update =  MLP(d,d,multiple_factor,dropout)    
        self.conv = GATv2Conv(d,d//heads,heads,edge_dim=d_type,dropout=dropout)
    
    def forward(self, x, edge_index, edge_attr):
        x_new = self.conv(x, edge_index, edge_attr)
        x_new = self.v_update(x_new)
        return x+x_new
    
    def __repr__(self):
        return 'GAT_block'  

class GNN_loss():
    def save_idx(self,train_idx,val_idx,y,y_val):
        self.register_buffer('train_idx', train_idx)
        self.val_idx = val_idx
        self.register_buffer('train_y', y[train_idx])
        self.y_val = y_val
        
    def get_loss(self,yhat):
        train_loss = self.loss(yhat[self.train_idx],self.train_y)
        val_loss = f1_score(self.y_val,\
                            yhat[self.val_idx].detach().cpu().numpy().argmax(1),\
                            average='micro')
        return train_loss,val_loss

class GNN1(torch.nn.Module,GNN_loss):
    def __init__(self,layers,d,d_type,in_d,out_d,node_embed_types,
                 edge_embed_types,heads,dropout,train_idx,val_idx,y,y_val,multiple_factor=2):
        super(GNN1, self).__init__()
        self.node_type_embed = Embedding(node_embed_types, d_type)
        self.edge_type_embed = Embedding(edge_embed_types, d_type)
        self.input_linear = MLP(in_d+d_type,d,1,dropout)

        self.conv = nn.ModuleList([GAT_block(d,d_type,heads,dropout,multiple_factor) for _ in range(layers)])
        self.out_linear = MLP(d,out_d,multiple_factor,dropout)
        self.loss = CrossEntropyLoss()
        self.save_idx(train_idx,val_idx,y,y_val)

        
    def forward(self, x, edge_index, node_type,edge_type,IsTrain=False):
        torch.set_grad_enabled(IsTrain)
        node_embed = self.node_type_embed(node_type)
        edge_embed = self.edge_type_embed(edge_type)
        x = torch.cat([x,node_embed],1)
        x = self.input_linear(x)
        for conv in self.conv:
            x = conv(x, edge_index, edge_embed)
        x = x[:4278] # take only the movie nodes
        x = self.out_linear(x)
        if IsTrain:
            return self.get_loss(x)
        else:
            return x        

# =============================================================================
# GNN2
# =============================================================================

class bipartite_msg(torch.nn.Module):
    def __init__(self,d,dropout,multiple_factor):
        super(bipartite_msg, self).__init__()   
        self.in_linear = Linear(d,d)
        self.out_linear = MLP(2*d,d,multiple_factor,dropout)
    
    def forward(self, M, N, adj_m2n):
        # message from M to N
        # adj_m2n is SparseTensor of shape (n,m)
        M = self.in_linear(M)
        M2N = adj_m2n.matmul(M)
        N_new = self.out_linear(torch.concat([N,M2N],1))
        return N + N_new

class bipartite_block(torch.nn.Module):
    def __init__(self,d,dropout,multiple_factor):
        super(bipartite_block, self).__init__()   
        self.m2a = bipartite_msg(d,dropout,multiple_factor)
        self.m2d = bipartite_msg(d,dropout,multiple_factor)
        self.d2m = bipartite_msg(d,dropout,multiple_factor)
        self.a2m = bipartite_msg(d,dropout,multiple_factor)
    
    def forward(self, M, A, D, adj_m2a, adj_m2d, adj_d2m, adj_a2m):
        A = self.m2a(M,A,adj_m2a)
        D = self.m2d(M,D,adj_m2d)
        M = self.d2m(D,M,adj_d2m)
        M = self.a2m(A,M,adj_a2m)
        return M, A, D


class GNN2(torch.nn.Module,GNN_loss):
    def __init__(self,layers,d,in_d,out_d,
                 dropout,train_idx,val_idx,y,y_val,multiple_factor=2):
        super(GNN2, self).__init__()

        self.input_linear_A = MLP(in_d,d,multiple_factor)
        self.input_linear_D = MLP(in_d,d,multiple_factor)
        self.input_linear_M = MLP(in_d,d,multiple_factor)
        self.conv = nn.ModuleList([bipartite_block(d,dropout,multiple_factor) for _ in range(layers)])
        self.out_linear = MLP(d,out_d,multiple_factor)

        self.loss = CrossEntropyLoss()
        self.save_idx(train_idx,val_idx,y,y_val)

        
    def forward(self,M,A,D,adj_m2d,adj_m2a,adj_d2m,adj_a2m,IsTrain=False):
        torch.set_grad_enabled(IsTrain)
        M = self.input_linear_M(M)
        A = self.input_linear_A(A)
        D = self.input_linear_D(D)
        for conv in self.conv:
            M, A, D = conv(M, A, D, adj_m2a, adj_m2d, adj_d2m, adj_a2m)
        x = self.out_linear(M)
        if IsTrain:
            return self.get_loss(x)
        else:
            return x        

def train_eval(model,data,epochs,y_val,val_idx,print_freq=1):
    # train
    opt = Adam(model.parameters())
    increase_count = 1
    lossBest = -1
    count = 0
    opt.zero_grad()
    for epoch in range(epochs):
        model.train()
        train_loss,val_loss = model(*data,IsTrain=True)
        if epoch>0 and epoch%print_freq==0:
            print("epoch:{}, train:{}, F1-val:{}".format(epoch,train_loss,val_loss))
        if val_loss>lossBest:
            lossBest = val_loss
            bestWeight = copy.deepcopy(model.state_dict())
            count = 0
        else:
            count += 1
            if count > increase_count:
                model.load_state_dict(bestWeight)
                break
        train_loss.backward()
        opt.step()
        opt.zero_grad()        
            
            
    # eval
    # no need to model.eval(). The batch stats is always better than running stats as we use the whole graph,
    # alternative is to set tracking_running_stats in BatchNorm to False.
    model.eval()
    with torch.no_grad():
        yhat = model(*data,IsTrain=False)
    yhat = yhat.detach().cpu().numpy().argmax(1)
    micro,macro = f1_score(y_val,yhat[val_idx],average='micro'),\
                    f1_score(y_val,yhat[val_idx],average='macro')
    if print_freq==1:
        print('micro:{}, macro:{}'.format(micro,macro))
    return {'micro':micro,'macro':macro}
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        