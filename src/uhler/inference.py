import torch
import numpy as np
import pickle
from argparse import Namespace
from utils import get_data


def evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode='CMVAE'):

	model = model.to(device)
	model.eval()
	
	gt_y = []
	pred_y = []
	c_y = []
	gt_x = []
	mu = []
	var = []

	for i, X in enumerate(dataloader):
		
		x = X[0]
		y = X[1]
		c = X[2]
		x = x.to(device)
		c = c.to(device)
		
		if numint == 2:
			idx = torch.nonzero(torch.sum(c, axis=0), as_tuple=True)[0]
			c1 = torch.zeros_like(c).to(device)
			c1[:,idx[0]] = 1
			c2 = torch.zeros_like(c).to(device)
			c2[:,idx[1]] = 1
		
		with torch.no_grad():

			if mode=='CMVAE':
				if numint == 1:
					y_hat, x_recon, z_mu, z_var, G = model(x, c, c, num_interv=1, temp=temp)
				else: 
					y_hat, x_recon, z_mu, z_var, G = model(x, c1, c2, num_interv=2, temp=temp)	

		gt_x.append(x.cpu().numpy())
		gt_y.append(y.numpy())
		pred_y.append(y_hat.detach().cpu().numpy())
		c_y.append(c.cpu().numpy())
		mu.append(z_mu.detach().cpu().numpy())
		var.append(z_var.detach().cpu().numpy())

	gt_x = np.vstack(gt_x)
	gt_y = np.vstack(gt_y)
	pred_y = np.vstack(pred_y)
	c_y = np.vstack(c_y)
	mu = np.vstack(mu)
	var = np.vstack(var)

	rmse = np.sqrt(np.mean(((pred_y[:] - gt_y[:])**2)) / np.mean(((gt_y[:])**2)))
	signerr = (np.sum(np.sum((np.sign(pred_y) != np.sign(gt_y)))) / gt_y.size)

	return rmse, signerr, gt_y, pred_y, c_y, gt_x, mu, var


def evaluate_single_leftout(model, path_to_dataloader, device, mode, temp=1):
	with open(f'{path_to_dataloader}/test_data_single_node.pkl', 'rb') as f:
		dataloader = pickle.load(f)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode=mode)

def evaluate_single_train(model, path_to_dataloader, device, mode, temp=1):
	with open(f'{path_to_dataloader}/train_data.pkl', 'rb') as f:
		dataloader = pickle.load(f)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode=mode)


def evaluate_double(model, path_to_ptbtargets, device, mode, temp=1):
	with open(f'{path_to_ptbtargets}/ptb_targets.pkl', 'rb') as f:
		ptb_targets = pickle.load(f)
	##
	_, dataloader, _, _, _ = get_data(mode='test', perturb_targets=ptb_targets)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=2, mode=mode)

