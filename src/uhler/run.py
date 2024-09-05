import os
import argparse
from argparse import Namespace
import time
import json
import pickle
import torch
import numpy as np
import random
from train import train
from utils import get_data

def main(args):
	print(f'using device: {args.device}')

	opts = Namespace(
		batch_size = 32,
		mode = 'train',
		lr = 1e-3,
		epochs = args.epochs,
		grad_clip = False,
		mxAlpha = 1,
		mxBeta = 1,
		mxTemp = 100,
		lmbda = 0.1,
		MMD_sigma = 200,
		kernel_num = 10,
		matched_IO = False,
		latdim = args.latdim, #our assumption 16
		seed = args.seed,
		trainmode = args.trainmode
	)

	torch.manual_seed(opts.seed)
	np.random.seed(opts.seed)
	random.seed(opts.seed)

	adata, dataloader, dataloader2, dim, cdim, ptb_targets = get_data(batch_size=opts.batch_size, mode=opts.mode)

	opts.dim = dim
	if opts.latdim is None:
		opts.latdim = cdim
	opts.cdim = cdim

	with open(f'{args.savedir}/config.json', 'w') as f:
		json.dump(opts.__dict__, f, indent=4)

	with open(f'{args.savedir}/ptb_targets.pkl', 'wb') as f:
		pickle.dump(ptb_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(f'{args.savedir}/test_data_single_node.pkl', 'wb') as f:
		pickle.dump(dataloader2, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(f'{args.savedir}/train_data.pkl', 'wb') as f:
		pickle.dump(dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)

	if args.model == 'cmvae':
		train(dataloader, opts, args.device, args.savedir, log=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parse args')
	parser.add_argument('-s', '--savedir', type=str, default='./../../result/uhler/', help='directory to save the results')
	parser.add_argument('--device', type=str, default='cuda:0', help='device to run the training')
	parser.add_argument('--model', type=str, default='cmvae', help='model to run the training')
	parser.add_argument('--name', type=str, default=f'full_go', help='name of the run')
	parser.add_argument('--trainmode', type = str, default = 'regular')
	parser.add_argument('--latdim', type = int, default = 70)
	parser.add_argument('--seed', type = int, default = 42)
	parser.add_argument('--epochs', type=int, default = 100)
	args = parser.parse_args()
	
	#concat
	args.name = f'{args.name}_{args.trainmode}'

	args.savedir = os.path.join(args.savedir, args.name)
	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

	#create folder for seed
	args.savedir = os.path.join(args.savedir, f'seed_{args.seed}')
	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

	main(args)
