
import math

import cv2
import torch
import numpy as np


def Rx(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Rx = torch.zeros((batchsize,3,3))
    Rx[:,0,0] = 1
    Rx[:,0,1] = 0
    Rx[:,0,2] = 0
    Rx[:,1,0] = 0
    Rx[:,1,1] = cosx
    Rx[:,1,2] = -sinx
    Rx[:,2,0] = 0
    Rx[:,2,1] = sinx
    Rx[:,2,2] = cosx
    return Rx

def Ry(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Ry = torch.zeros((batchsize,3,3))
    Ry[:,0,0] = cosx
    Ry[:,0,1] = 0
    Ry[:,0,2] = sinx
    Ry[:,1,0] = 0
    Ry[:,1,1] = 1
    Ry[:,1,2] = 0
    Ry[:,2,0] = -sinx
    Ry[:,2,1] = 0
    Ry[:,2,2] = cosx
    return Ry

def Rz(x):
    batchsize = x.shape[0]
    sinx = torch.sin(x)
    cosx = torch.cos(x)
    Rz = torch.zeros((batchsize,3,3))
    Rz[:,0,0] = cosx
    Rz[:,0,1] = -sinx
    Rz[:,0,2] = 0
    Rz[:,1,0] = sinx
    Rz[:,1,1] = cosx
    Rz[:,1,2] = 0
    Rz[:,2,0] = 0
    Rz[:,2,1] = 0
    Rz[:,2,2] = 1
    return Rz

def R(euler):
    rx = Rx(euler[:,0])
    ry = Ry(euler[:,1])
    rz = Rz(euler[:,2])

    return torch.bmm(rx,torch.bmm(ry,rz))


# euler angles in x,y,z
def euler2rotm(euler):

    x,y,z = euler
    Rx = np.array([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
    Ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
    Rz = np.array([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])
    return Rx @ Ry @ Rz

# create 3DMM using alphas for shape eigen vectors, and betas for expression eigen vectors
def create3DMM(mu_s, mu_exp, s_eigen, exp_eigen, alphas, betas):
    shape_cov = torch.matmul(s_eigen,alphas)
    exp_cov = torch.matmul(exp_eigen,betas)

    shape = (mu_s + shape_cov.view((53215,3))) + (mu_exp + exp_cov.view((53215,3)))
    return shape

# rotate and translate the shape according to rotation translation and scale factor
def align(shape,s,R,T):
    return s*(torch.matmul(shape,R) + T)

# apply orthographic projection
def project(shape):
    ortho = torch.Tensor([[1,0,0],[0,1,0]]).float().cuda()
    return torch.matmul(shape,ortho.T)

# predict a 3DMM model according to parameters
def predict(s,R,T,alphas,betas):
    shape = create3DMM(alphas,betas)
    shape = align(shape,s,R,T)
    shape = project(shape)

    return shape

def viewLM(myimg,lm2d):
    myimg = cv2.cvtColor(myimg,cv2.COLOR_BGR2RGB)
    for p in lm2d:
        x = int(p[0])
        y = int(p[1])
        cv2.circle(myimg,(x,y),2,[0,0,255],-1)

    cv2.imshow('img', (myimg).astype(np.uint8))
    cv2.waitKey(1)

    return myimg

if __name__ == '__main__':

    import dataloader

    facemodel = dataloader.Face3DMM()

    mu_s = facemodel.mu_shape
    mu_exp = facemodel.mu_exp
    s_eigen = facemodel.shape_eigenvec
    exp_eigen = facemodel.exp_eigenvec
    lm = facemodel.lm

    alphas  = torch.matmul(torch.randn(199),torch.eye(199)).float().cuda()
    betas = torch.matmul(torch.randn(29),torch.eye(29)).float().cuda()

    euler = np.random.rand(3)
    R = torch.Tensor(euler2rotm(euler)).float().cuda()
    T = torch.randn((1,3)).float().cuda() * 10
    s = torch.randn(1).float().cuda()

    shape = create3DMM(mu_s,mu_exp,s_eigen,exp_eigen,alphas,betas)
    shape = align(shape,s,R,T)
    shape = project(shape)

    keypoints = shape[lm,:]
    print(shape.shape)
    print(keypoints.shape)

    pts = keypoints.detach().cpu().numpy()
    print(pts.shape)

    import scipy.io
    scipy.io.savemat('pts.mat',{'pts': pts})
