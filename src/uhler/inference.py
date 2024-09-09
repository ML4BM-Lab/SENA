import torch
import numpy as np
import pickle
from argparse import Namespace
from utils import get_data
from train import loss_function
from utils import MMD_loss, load_norman_2019_dataset, SCDATA_sampler, SCDataset, DataLoader
from collections import defaultdict


def evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode='CMVAE', MMD_sigma=200, kernel_num=10):

	model = model.to(device)
	model.eval()
	
	pred_x, gt_x = [], []
	gt_y, pred_y = [], []
	c_y, mu, var = [], [], []

	#ll = defaultdict(int)
	for i, X in enumerate(dataloader):
		
		x = X[0]
		y = X[1]
		c = X[2]
		x = x.to(device)
		c = c.to(device)

		# for kk in c.argmax(axis=1):
		# 	ll[kk.__int__()] += 1
		
		if numint == 2:
			idx = torch.nonzero(torch.sum(c, axis=0), as_tuple=True)[0]
			c1 = torch.zeros_like(c).to(device)
			c1[:,idx[0]] = 1
			c2 = torch.zeros_like(c).to(device)
			c2[:,idx[1]] = 1
		
		with torch.no_grad():

			if mode=='CMVAE':
				if numint == 1:
					y_hat, x_recon, z_mu, z_var, G, _ = model(x, c, c, num_interv=1, temp=temp)
				else: 
					y_hat, x_recon, z_mu, z_var, G, _ = model(x, c1, c2, num_interv=2, temp=temp)	

		gt_x.append(x.cpu())
		pred_x.append(x_recon.cpu())

		gt_y.append(y)
		pred_y.append(y_hat.detach().cpu())

		c_y.append(c.cpu())
		mu.append(z_mu.detach().cpu())
		var.append(z_var.detach().cpu())

	#stack
	gt_x = torch.vstack(gt_x)
	pred_x = torch.vstack(pred_x)
	gt_y = torch.vstack(gt_y)
	pred_y = torch.vstack(pred_y)
	c_y = torch.vstack(c_y)
	mu = torch.vstack(mu)
	var = torch.vstack(var)
	G = model.G.detach().cpu()

	#compute metrics
	_, MSE, KLD, L1 = loss_function(pred_y, gt_y, pred_x, gt_x, mu, var, G , MMD_sigma=MMD_sigma, kernel_num=kernel_num, matched_IO=True)

	#compute MMD by batches
	mmd_loss = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
	bs, MMD = 16, []
	for i in range(int(pred_y.shape[0]//bs)):
		MMD.append(mmd_loss(pred_y[i*bs:(i+1)*bs], gt_y[i*bs:(i+1)*bs]).__float__())
	MMD = np.mean(MMD)

	return MMD, MSE.__float__(), KLD.__float__(), L1.__float__()


def evaluate_single_leftout(model, path_to_dataloader, device, mode, temp=1000):
	with open(f'{path_to_dataloader}/test_data_single_node.pkl', 'rb') as f:
		dataloader = pickle.load(f)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode=mode)

def evaluate_single_train(model, path_to_dataloader, device, mode, temp=1000):
	with open(f'{path_to_dataloader}/train_data.pkl', 'rb') as f:
		dataloader = pickle.load(f)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode=mode)


def evaluate_double(model, path_to_ptbtargets, device, mode, temp=1000):

	#load perturbation targets
	with open(f'{path_to_ptbtargets}/ptb_targets.pkl', 'rb') as f:
		ptb_targets = pickle.load(f)

	dataloader, _, _, _ = get_data(mode='test', perturb_targets=ptb_targets)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=2, mode=mode)

