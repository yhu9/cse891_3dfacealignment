

import sys

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
model.eval()

# read the adjustment network checkpoint
adjustmentnet.load_state_dict(torch.load("model/chkpoint_adj_000.pt"))
adjustmentnet.cuda()
adjustmentnet.eval()

# read the img file from system argument
filenamein = sys.argv[1]
img = io.imread(filenamein)

# reshape image to 224x224
img = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
img = torch.Tensor(img).float().cuda()
img = img.permute(2,0,1).unsqueeze(0)

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
shape_cov = torch.bmm(torch.stack(batchsize*[face3dmm.shape_eigenvec]),alpha_matrix)
exp_cov = torch.bmm(torch.stack(batchsize*[face3dmm.exp_eigenvec]),beta_matrix)
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










