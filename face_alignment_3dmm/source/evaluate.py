import sys

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

# read the model checkpoint
model = densenet
model.load_state_dict(torch.load("model/chkpoint_000.pt"))
model.cuda()

# read the adjustment network checkpoint
adjustmentnet.load_state_dict(torch.load("model/chkpoint_adj_000.pt"))
adjustmentnet.cuda()

# create the loader
loader = dataloader.W300Loader()

for i, batch in enumerate(loader):
    img = batch['image'].cuda().unsqueeze(0)
    lm_gt = batch['lm2d'].cpu().data.numpy()
    lm_gt = lm_gt*112+112

    # apply model on input
    y_pred = model(img)
    alphas = y_pred[:,:199]
    betas = y_pred[:,199:228]
    s = y_pred[:,228]
    t = torch.tanh(y_pred[:,229:231])
    r = torch.tanh(y_pred[:,231:235]) * (3.14/4)

    # create 3dmm
    alpha_matrix = alphas.unsqueeze(2).expand(*alphas.size(),alphas.size(1)) * torch.eye(alphas.size(1)).cuda()
    beta_matrix = betas.unsqueeze(2).expand(*betas.size(),betas.size(1))*torch.eye(betas.size(1)).cuda()
    shape_cov = torch.bmm(torch.stack([face3dmm.shape_eigenvec]),alpha_matrix)
    exp_cov = torch.bmm(torch.stack([face3dmm.exp_eigenvec]),beta_matrix)
    shape_cov = shape_cov.sum(2)
    exp_cov = exp_cov.sum(2)

    shape = (face3dmm.mu_shape.unsqueeze(0) + shape_cov.view((1,53215,3))) + exp_cov.view((1,53215,3))
    lm = shape[:,face3dmm.lm,:]
    R = util.R(r).cuda()
    scaledshape = s.unsqueeze(1).unsqueeze(1)*torch.bmm(lm,R)
    alignedshape = t.unsqueeze(1) + scaledshape[:,:,:2]

    # adjustment network applied onto the landmarks of 3dMM taking input image
    adjustment = torch.tanh(adjustmentnet(img).view(-1,68,2))
    output = alignedshape + adjustment

    pred = output.cpu().data.numpy()
    pred = pred*112 + 112
    pred = pred[0]

    le = lm_gt[36,:]
    re = lm_gt[45,:]
    d = np.linalg.norm(le - re)

    diff = pred - lm_gt
    mse = np.mean(np.linalg.norm(diff,1))
    NME = mse / d

    sample = img.squeeze().cpu().permute(1,2,0).data.numpy()
    sample = sample*255

    util.viewLM(sample.astype(np.uint8),pred)

