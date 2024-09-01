import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import scanpy as sc
import pickle
import torch
from inference import evaluate_single_leftout, evaluate_double
import pandas as pd 
import os
import seaborn as sns
import graphical_models as gm
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from utils import get_data
from collections import defaultdict

#load model
mode = 'regular'
subsample = 'full_go'
seed = 13
savedir = os.path.join('./../../../','result','uhler',f'{subsample}_{mode}',f'seed_{seed}')
model = torch.load(f'{savedir}/best_model.pt')

#load dataset
device = "cuda:0"
_, dataloader, dataloader2, dim, cdim, ptb_targets = get_data(batch_size=1, mode='train')

#iterate
intervention_exp = defaultdict(lambda: defaultdict(list))
for X in tqdm(dataloader):
    
    x = X[0]
    y = X[1]
    c = X[2]
    
    if model.cuda:
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)

    #get index
    idx = np.where(c.detach().cpu().numpy())[1][0]

    #"""get intervention arrays"""

    # decode an interventional sample from an observational sample    
    bc, csz = model.c_encode(c, temp=1)  
    # bc2, csz2 = model.c_encode(c, temp=1)     
    intervention_exp[idx]['int'].append((bc * csz.reshape(-1,1)).detach().cpu().numpy())
    intervention_exp[idx]['int_raw'].append((bc).detach().cpu().numpy())

    #"""z"""

    #get z
    mu, var = model.encode(x)
    z = model.reparametrize(mu, var)
    intervention_exp[idx]['z'].append((z).detach().cpu().numpy())

    #"""u"""

    #get u
    u = model.dag(z, bc, csz, bc, csz, num_interv=1)
    intervention_exp[idx]['u'].append((u).detach().cpu().numpy())

def study_repr_vectors(varname = 'int'):

    """explore repr vectors"""
    ninterv = 70
    assert sorted(intervention_exp.keys()) == list(range(ninterv))
    repr_mat = np.zeros(shape=(ninterv,ninterv))

    for idx in intervention_exp:

        # max_arg = np.argmax(np.abs(np.vstack(intervention_exp[idx][var]).mean(axis=0)))
        # print(max_arg)
        repr_mat[idx,:] = np.vstack(intervention_exp[idx][varname]).mean(axis=0)

    repr_mat_normrow = repr_mat#MinMaxScaler().fit_transform(repr_mat)

    # Create a figure and axis
    plt.figure(figsize=(12, 12))

    # Create the heatmap
    sns.heatmap(repr_mat_normrow, 
                annot=False,       # Annotate cells with their values
                fmt=".2f",        # Format the annotations to 2 decimal places
                cmap="coolwarm",  # Use a visually appealing color palette
                cbar=True,        # Show the color bar
                square=True,      # Ensure the cells are square-shaped
                linewidths=0.5,   # Add lines between cells for better readability
                linecolor='white' # Color of the lines between cells
            )

    # Add titles and labels
    plt.title('Heatmap of Square Matrix', fontsize=16)
    plt.xlabel('Column Index', fontsize=14)
    plt.ylabel('Row Index', fontsize=14)

    # Adjust layout to ensure everything fits well
    plt.tight_layout()

    # Display the heatmap
    plt.savefig(os.path.join('./../../','figures','uhler',f'{subsample}_{mode}','causal_graph',f'heatmap_repr_{varname}.png'))


##
study_repr_vectors(varname = 'int')
study_repr_vectors(varname = 'int_raw')
study_repr_vectors(varname = 'z')
study_repr_vectors(varname = 'u')