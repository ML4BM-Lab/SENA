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
## pygraphviz -> sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config   

def draw(pdag, colored_set=set(), solved_set=set(), affected_set=set(), nw_ax=None, edge_weights=None, node_label=None):

    """ 
    plot a partially directed graph
    """
    plt.clf()

    p = pdag.nnodes

    if nw_ax is None:
        nw_ax = plt.subplot2grid((4, 4), (0, 0), colspan=12, rowspan=12)

    plt.gcf().set_size_inches(20, 20)

    # directed edges
    d = nx.DiGraph()
    d.add_nodes_from(list(range(p)))
    for (i, j) in pdag.arcs:
        d.add_edge(i, j)

    # undirected edges
    e = nx.Graph()
    try:
        for pair in pdag.edges:
            (i, j) = tuple(pair, length = 10)
            e.add_edge(i, j)
    except:
        print('there are no undirected edges')
    
    # edge color
    if edge_weights is not None:
        color_d = []
        for i,j in d.edges:
            color_d.append(edge_weights[i,j])

        color_e = []
        for i,j in e.edges:
            color_e.append(edge_weights[i, j])
    else:
        color_d = 'k'
        color_e = 'k'


    # plot
    print("plotting...")
    # pos = nx.circular_layout(d)
    pos = graphviz_layout(d, prog='dot')
    nx.draw_networkx(e, pos=pos, node_color='w', style = 'dashed',  edge_cmap=plt.cm.Blues, edge_vmin=-0.025, edge_vmax=0.025, edge_color=color_e)
    color = ['w']*p
    for i in affected_set:
        color[i] = 'orange'
    for i in colored_set:
        color[i] = 'y'
    for i in solved_set:
        color[i] = 'grey'

    nx.draw_networkx(d, pos=pos, node_color=color, ax=nw_ax, edge_cmap=plt.cm.RdBu_r, edge_vmin=-0.025, edge_vmax=0.025, edge_color=color_d, width=1.5, with_labels=True)
    #nx.draw_networkx_labels(d, pos, labels={node: node_label[node] for node in range(p)}, ax=nw_ax, font_size=12.5)

    plt.savefig(os.path.join('./../../','result', model_name, f'{model_name}_DAG_graph.png'))
    plt.clf()
    plt.cla()
    plt.close()

def draw_spectrum(A, B, savefile=None):

    plt.clf()
    plt.figure(figsize=(3,3))
    e_A = np.linalg.eigvalsh(np.matmul(A.T, A))[::-1]
    e_B = np.linalg.eigvalsh(np.matmul(B.T, B))[::-1]
    plt.plot(np.maximum(e_A,0)**0.5, label=r'$(I-B)^{-1}$')
    plt.plot(np.maximum(e_B,0)**0.5, label=r'$B$')
    plt.legend()
    plt.ylabel('eigenvalues')
    plt.xlabel('index')
    plt.title('Spectrum of SCM')
    plt.tight_layout()
    
    if savefile is not None:
        plt.savefig(savefile)

    plt.show()
    plt.close()

def generate_DAG():

    savedir = f'./../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
        ptb_targets = pickle.load(f)

    #get DAG
    G_dense = torch.triu(model.G, diagonal=1).cpu().detach().numpy()

    ## plot it
    plt.figure(figsize=(15, 15))
    im = plt.imshow(G_dense, cmap=plt.cm.RdBu_r, vmin=-0.025, vmax=0.025)
    plt.grid(False)
    plt.xticks(range(G_dense.shape[0]), range(G_dense.shape[0]))
    plt.yticks(range(G_dense.shape[0]), range(G_dense.shape[0]))
    cb = plt.colorbar(im)
    cb.outline.set_visible(False)
    plt.title('G (no threshold)')
    plt.tight_layout()
    plt.savefig(os.path.join('./../../','result', model_name, f'{model_name}_DAG_dense.png'))

    ## plot the graph
    G = np.multiply(G_dense, abs(G_dense)>0.005)
    dag = gm.DAG(set(range(G_dense.shape[0])))
    for i in range(G_dense.shape[0]):
        for j in np.arange(i, G_dense.shape[0], 1):
            if abs(G[i,j])>0.005: ## apply th?
                dag.add_arc(i,j)

    node_label = [f'P{i}' for i in range(G_dense.shape[0])]
    draw(dag, edge_weights=G, node_label=node_label)

def generate_regulatory_programs():

    #load model
    savedir = f'./../../result/{model_name}' 
    model = torch.load(f'{savedir}/best_model.pt')

    with open(f'{savedir}/ptb_targets.pkl', 'rb') as f:
        ptb_targets = pickle.load(f)

    gene_c_mxidx = []
    gene_c_vec = []

    ##get the contribution of each ptb to each of the latent factors
    for i in tqdm(range(len(ptb_targets))):
        c = torch.zeros((1,len(ptb_targets)), dtype=torch.double)
        c[0][i] = 1
        c = c.to(model.device)
        gene_c_mxidx.append(model.c_encode(c, temp=20)[0].argmax().item())
        gene_c_vec.append(model.c_encode(c, temp=20)[0].detach()[0])

    gene_c_mat = np.array([g_vec.cpu().numpy() for g_vec in gene_c_vec])

    ##plot it
    plt.figure(figsize=(15, 5))
    im = plt.imshow(gene_c_mat.T[:,:], cmap=plt.cm.RdPu, vmin=0, vmax=1)
    plt.grid(False)

    cb = plt.colorbar(im)
    cb.outline.set_visible(False)
    plt.yticks(range(model.G.shape[0]), [f'Program {i}' for i in range(model.G.shape[0])])
    plt.xticks(range(len(ptb_targets)), np.array(ptb_targets)[:], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join('./../../','result', model_name, f'{model_name}_regulatory_programs.png'))

## model name
model_name = 'full_go_regular'