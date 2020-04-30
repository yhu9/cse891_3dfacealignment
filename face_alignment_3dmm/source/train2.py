
import itertools
import argparse
import os

import torch
import torch.optim
import numpy as np
from skimage import io, transform

from model import densenet
from model import adjustmentnet
import dataloader
import util

####################################################

def train():

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = densenet
    model.load_state_dict(torch.load("model/chkpoint_000.pt"))
    model.cuda()
    #optimizer2 = torch.optim.Adam(model.parameters(),lr=1e-4)

    adjustmentnet.load_state_dict(torch.load("model/chkpoint_adj_000.pt"))
    adjustmentnet.cuda()
    adjustmentnet.train()

    loader = dataloader.BatchLoader
    face3dmm = dataloader.Face3DMM()
    optimizer = torch.optim.Adam(list(adjustmentnet.parameters())+list(model.parameters()),lr=1e-5)

    # main training loop
    for epoch in itertools.count():
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            x = batch['image'].cuda()
            y = batch['lm2d'].cuda()
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
            pred = (alignedshape + adjustment) * 112 + 112
            gt = y * 112 + 112

            # weight update
            loss = torch.mean(torch.norm(gt - pred,p=2,dim=2))
            loss.backward()
            optimizer.step()

            gt = gt[0].cpu().data.numpy()
            pred = pred[0].cpu().data.numpy()
            sample = x[0].cpu().permute(1,2,0).data.numpy()
            sample = sample*255
            util.viewLM(sample.astype(np.uint8),pred)
            #io.imsave(f"example_{i:04d}.png",sample)

            print(f"epoch/batch {epoch}/{i}  |   Loss: {loss:.4f}")

        print("saving!")
        torch.save(model.state_dict(), f"model/chkpoint_{epoch:03d}.pt")
        torch.save(adjustmentnet.state_dict(), f"model/chkpoint_adj_{epoch:03d}.pt")

####################################################################################3
if __name__ == '__main__':
    train()




