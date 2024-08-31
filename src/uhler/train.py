import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from copy import deepcopy
import numpy as np
import os
import pandas as pd

from model import CMVAE
from utils import MMD_loss, compute_activation_df, compute_outlier_activation_analysis, load_norman_2019_dataset


# fit CMVAE to data
def train(
    dataloader,
    opts,
    device,
    savedir,
    log,
    order=None,
    nonlinear=False,
    ):

    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    cmvae = CMVAE(
        dim = opts.dim,
        z_dim = opts.latdim,
        c_dim = opts.cdim,
        device = device, 
        dataloader = dataloader,
        mode = opts.trainmode
    )

    cmvae.double()
    cmvae.to(device)

    optimizer = torch.optim.Adam(params=cmvae.parameters(), lr=opts.lr) #do not use Adam if sparse 

    cmvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))

    ## Loss parameters
    beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
    beta_schedule[:10] = 0
    beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
    alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
    alpha_schedule[:] = opts.mxAlpha
    alpha_schedule[:5] = 0
    alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
    alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

    ## Softmax temperature 
    temp_schedule = torch.ones(opts.epochs)
    temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)

    min_train_loss = np.inf
    results = []
    mode = opts.trainmode
    adata, ptb_targets, _, gos, _ = load_norman_2019_dataset()

    #best_model = deepcopy(cmvae)
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        L1Av = 0
        for (i, X) in enumerate(dataloader):
            
            x = X[0]
            y = X[1]
            c = X[2]
            
            if cmvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                
            optimizer.zero_grad()
            y_hat, x_recon, z_mu, z_var, G = cmvae(x, c, c, num_interv=1, temp=temp_schedule[n])
            mmd_loss, recon_loss, kl_loss, L1 = loss_function(y_hat, y, x_recon, x, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss + opts.lmbda*L1
            loss.backward()
            if opts.grad_clip:
                for param in cmvae.parameters():
                    print(param)
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()
            L1Av += L1.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})
                wandb.log({'l1_loss': L1})

        print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct)+', '+'L1='+str(L1Av/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})
            wandb.log({'epoch avg l1_loss': L1/ct})

        if (mmdAv + reconAv + klAv + L1Av)/ct < min_train_loss:
            min_train_loss = (mmdAv + reconAv + klAv + L1Av)/ct 
            #best_model = deepcopy(cmvae)
            torch.save(cmvae, os.path.join(savedir, 'best_model.pt'))

        ## report
        ttest_df = compute_activation_df(cmvae, adata, gos, scoretype = 'mu_diff', mode = mode)
        summary_analysis_ep = compute_outlier_activation_analysis(ttest_df, adata, ptb_targets, mode = mode)
        summary_analysis_ep['epoch'] = n
        summary_analysis_ep['mmd_loss'] = mmdAv/ct
        summary_analysis_ep['recon_loss'] = reconAv/ct
        summary_analysis_ep['kl_loss'] = klAv/ct
        summary_analysis_ep['l1_loss'] = (L1/ct).__float__()

        #append
        results.append(summary_analysis_ep)

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(savedir, f'uhler_{mode}_summary.tsv'),sep='\t')
    print(results_df)
    #last_model = deepcopy(cmvae)
    #torch.save(last_model, os.path.join(savedir, 'last_model.pt'))


# loss function definition
def loss_function(y_hat, y, x_recon, x, mu, var, G, MMD_sigma, kernel_num, matched_IO=False):

    if not matched_IO:
        matching_function_interv = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num) # MMD Distance since we don't have paired data
    else:
        matching_function_interv = nn.MSELoss() # MSE if there is matched interv/observ samples
    matching_function_recon = nn.MSELoss() # reconstruction

    if y_hat is None:
        MMD = 0
    else:
        MMD = matching_function_interv(y_hat, y)
    MSE = matching_function_recon(x_recon, x)
    logvar = torch.log(var)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)/x.shape[0]
    if G is None:
        L1 = 0
    else:
        L1 = torch.norm(torch.triu(G,diagonal=1),1)  # L1 norm for sparse G
    return MMD, MSE, KLD, L1



