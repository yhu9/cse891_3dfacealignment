
import itertools
import argparse

import torch
import torch.optim

from model import densenet
from model import adjustmentnet
import dataloader
import util

####################################################

def train():

    # define model, dataloader, 3dmm eigenvectors, optimization method
    model = densenet
    model.cuda()
    adjustmentnet.cuda()
    loader = dataloader.BatchLoader
    face3dmm = dataloader.Face3DMM()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

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
            # adjustment = adjustmentnet(x).view(-1,68,2)

            # weight update
            loss = torch.mean(torch.abs(y - (alignedshape)))
            loss.backward()
            optimizer.step()

            print(f"epoch/batch {epoch}/{i}  |   Loss: {loss:.4f}")
        print("saving!")
        torch.save(model.state_dict(), f"model/chkpoint_{epoch:03d}.pt")


####################################################################################3
if __name__ == '__main__':
    train()

