import sys
import itertools
import argparse
import os

import numpy as np
import torch
import cv2
from skimage import io, transform

import dataloader
import util
from model import densenet
from model import adjustmentnet

# read the 3dmm eigenvectors
face3dmm = dataloader.Face3DMM()

# define model, dataloader, 3dmm eigenvectors, optimization method
model = densenet
model.load_state_dict(torch.load("model/chkpoint_000.pt"))
model.cuda()

adjustmentnet.load_state_dict(torch.load("model/chkpoint_adj_000.pt"))
adjustmentnet.cuda()

loader = dataloader.W300Loader()
#loader = dataloader.LFWLoader()
#loader = dataloader.BatchLoader
face3dmm = dataloader.Face3DMM()

results = []
# main training loop
for i, batch in enumerate(loader):
    x = batch['image'].cuda().unsqueeze(0)
    y = batch['lm2d'].cuda().unsqueeze(0)
    y_pred = model(x)
    batchsize = x.shape[0]

    alphas = y_pred[:,:199]
    betas = y_pred[:,199:228]
    s = y_pred[:,228]
    t = y_pred[:,229:231]
    t = torch.tanh(t)
    r = y_pred[:,231:235]
    r = torch.tanh(r) * (3.14/4)

    # apply 3DMM model from predicted parameters
    alpha_matrix = alphas.unsqueeze(2).expand(*alphas.size(),alphas.size(1)) * torch.eye(alphas.size(1)).cuda()
    beta_matrix = betas.unsqueeze(2).expand(*betas.size(),betas.size(1))*torch.eye(betas.size(1)).cuda()
    shape_cov = torch.bmm(torch.stack(batchsize*[face3dmm.shape_eigenvec]),alpha_matrix)
    exp_cov = torch.bmm(torch.stack(batchsize*[face3dmm.exp_eigenvec]),beta_matrix)
    shape_cov = shape_cov.sum(2)
    exp_cov = exp_cov.sum(2)

    # alignment
    shape = (face3dmm.mu_shape.unsqueeze(0) + shape_cov.view((batchsize,53215,3))) + exp_cov.view((batchsize,53215,3))
    lm = shape[:,face3dmm.lm,:]
    R = util.R(r).cuda()
    scaledshape = s.unsqueeze(1).unsqueeze(1)*torch.bmm(lm,R)
    alignedshape = t.unsqueeze(1) + scaledshape[:,:,:2]

    # adjustment network applied onto the landmarks of 3dMM taking input image
    adjustment = torch.tanh(adjustmentnet(x).view(-1,68,2))
    pred = adjustment* 112 + 112
    gt = y * 112 + 112

    # weight update
    gt = gt[0].cpu().data.numpy()
    pred = pred[0].cpu().data.numpy()

    le = gt[36,:]
    re = gt[45,:]
    d = np.linalg.norm(le - re)

    diff = pred - gt
    mse = np.mean(np.linalg.norm(diff,1))
    NME = mse / d

    # visualize results
    results.append(NME)
    print(i,NME)
    sample = x[0].cpu().permute(1,2,0).data.numpy()
    sample = (sample*255).astype(np.uint8)
    sample = util.viewLM(sample,pred)
    io.imsave(f"example_{i:04d}.png",sample)


idx = results.index(min(results))
print(idx, min(results))
avg_result = np.mean(results)
print(f"avg NME inter ocular: {avg_result}")

