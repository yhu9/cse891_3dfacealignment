
import os

import torch
from skimage import io, transform
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# the main data class which has iterators to all datasets
# add more datasets accordingly

# loader for testing dataset
class W300Loader(Dataset):

    def __init__(self,mode="indoor"):
        rootdir = "/home/huynshen/Downloads/300w/300w/300W"
        indoordir = os.path.join(rootdir,'01_Indoor')
        outdoordir = os.path.join(rootdir,'02_Outdoor')


        self.transform_img = transforms.Compose([Rescale((224,224)),ToTensor()])
        self.transform_lm = transforms.Compose([ToTensor()])
        self.mode = mode
        self.indoorfiles = [os.path.join(indoordir,f"indoor_{i:03d}.png") for i in range(1,301)]
        self.outdoorfiles = [os.path.join(outdoordir,f"outdoor_{i:03d}.png") for i in range(1,301)]
        self.indoorpts = [os.path.join(indoordir,f"indoor_{i:03d}.pts") for i in range(1,301)]
        self.outdoorpts = [os.path.join(outdoordir,f"outdoor_{i:03d}.pts") for i in range(1,301)]

    def __len__(self):
        return len(self.indoofiles) + len(self.outdoorfiles)

    def readptsfile(self,file):
        with open(file,'r') as fin:
            txt = fin.read()
            for i in range(3):
                idx = txt.index('\n')
                txt = txt[idx+1:]
            txt = txt[:-3]
        data = np.fromstring(txt,sep=' ')
        return data.reshape((68,2))

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == 'indoor':
            imgfile = self.indoorfiles[idx]
            ptsfile = self.indoorpts[idx]
        if self.mode == 'outdoor':
            imgfile = self.outdoorfiles[idx]
            ptsfile = self.outdoorpts[idx]

        img = io.imread(imgfile)
        pts = self.readptsfile(ptsfile)

        le = np.mean(pts[36:42,:],axis=0)
        re = np.mean(pts[42:48,:],axis=0)
        center = (le+re)/2
        h,w = img.shape[:2]

        # centralize eyes in test image
        minw = np.min(pts[:,0])
        maxw = np.max(pts[:,0])
        minh = np.min(pts[:,1])
        maxh = np.max(pts[:,1])
        width=int(2*(maxw - minw))
        height=int(2*(maxh - minh))
        length = max(width,height)
        canvas = np.zeros((length,length,3))

        minx = int(center[0] - length//2)
        maxx = int(center[0] + length//2)
        miny = int(center[1] - length//2)
        maxy = int(center[1] + length//2)

        top_img = np.maximum(0,miny).astype(np.int)
        bot_img = np.minimum(maxy,h).astype(np.int)
        left_img = np.maximum(0,minx).astype(np.int)
        right_img = np.minimum(maxx,w).astype(np.int)

        top = int(-miny) if miny < 0 else 0
        left = int(-minx) if minx < 0 else 0
        bot = length + (h-maxy) if (h-maxy) < 0 else length
        right =  length + (w-maxx) if (w-maxx) < 0 else length

        h1 = bot - top
        w1 = right - left
        h2 = bot_img - top_img
        w2 = right_img - left_img

        if h1 < h2: bot_img = int(bot_img - (h2 - h1))
        if h1 > h2: top = int(top + (h1 - h2))
        if w1 < w2: right_img = int(right_img - (w2 - w1))
        if w1 > w2: left = left + int((w1-w2))

        canvas[top:bot,left:right,:] = img[top_img:bot_img,left_img:right_img,:]

        img = self.transform_img(canvas) / 255


        # adjust landmarks to centralized face
        pts[:,0] = pts[:,0] - minw
        pts[:,1] = pts[:,1] - minh
        h,w,d = canvas.shape
        newd,newh,neww = img.shape
        pts[:,0] = (pts[:,0] / w) * neww
        pts[:,1] = (pts[:,1] / h) * newh
        pts[:,0] = pts[:,0] - (neww // 2)
        pts[:,1] = pts[:,1] - (newh // 2)
        pts = pts / 112
        pts = self.transform_lm(pts)

        sample = {'image': img, 'lm2d': pts}

        return sample

# LOADER FOR training dataset
class LFWLoader(Dataset):

    def __init__(self,
            root_dir = "/home/huynshen/data/face_alignment/300W_LP"
            ):

        self.transform_img = transforms.Compose([Rescale((224,224)),ToTensor()])
        self.transform_lm = transforms.Compose([ToTensor()])
        self.img_dir = root_dir
        self.imgfiles = []
        self.matfiles = []
        datasets = ["AFW","AFW_Flip","HELEN","HELEN_Flip","IBUG","IBUG_Flip","LFPW","LFPW_Flip"]
        for datadir in datasets:
            full_path = os.path.join(root_dir,datadir)
            files = os.listdir(full_path)
            files.sort()
            imgfiles = [os.path.join(full_path,f) for f in files if f[-3:] == 'jpg']
            matfiles = [os.path.join(full_path,f) for f in files if f[-3:] == 'mat']

            self.imgfiles = self.imgfiles + imgfiles
            self.matfiles = self.matfiles + matfiles

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get path
        imgpath = self.imgfiles[idx]
        matpath = self.matfiles[idx]

        # get data
        data = scipy.io.loadmat(matpath)
        lm2d = data['pt2d']
        img = io.imread(imgpath)
        img = img / 255.0
        h,w,d = img.shape

        # apply transforms
        img = self.transform_img(img)
        lm2d = self.transform_lm(lm2d)

        # determine keypoints in resized image
        new_d, new_h, new_w = img.shape
        lm2d[0,:] = (lm2d[0,:]/w)*new_w
        lm2d[1,:] = (lm2d[1,:]/h)*new_h
        lm2d = lm2d.T
        lm2d[:,0] = lm2d[:,0] - (new_w // 2)
        lm2d[:,1] = lm2d[:,1] - (new_h // 2)
        lm2d = lm2d / 112

        sample = {'image': img,'lm2d':lm2d}
        return sample

# LOADER FOR BIWI KINECT DATASET ONLY
class Face3DMM():

    def __init__(self,
            root_dir = "/home/huynshen/data/face_alignment/300W_LP/Code/ModelGeneration"
            ):

        shape_path = os.path.join(root_dir,"Model_Shape.mat")
        exp_path = os.path.join(root_dir,"Model_Exp.mat")
        self.transform = transforms.Compose([ToTensor()])

        # load shape data
        shape_data = scipy.io.loadmat(shape_path)
        mu_shape = shape_data['mu_shape']
        shape_eigenvec = shape_data['shape_eigenvec']
        self.lm = torch.from_numpy(shape_data['keypoints'].astype(np.int32)).long().cuda().squeeze()
        self.mu_shape = torch.from_numpy(mu_shape.reshape(53215,3)).float().cuda()
        self.mu_shape = self.mu_shape - self.mu_shape.mean(0).unsqueeze(0)
        self.mu_shape = self.mu_shape / 10000
        self.shape_eigenvec = torch.from_numpy(shape_eigenvec).float().cuda()

        # load expression data
        exp_data = scipy.io.loadmat(exp_path)
        mu_exp = exp_data['mu_exp']
        exp_eigenvec = exp_data['w_exp']

        self.mu_exp = torch.from_numpy(mu_exp.reshape(53215,3)).float().cuda()
        self.exp_eigenvec = torch.from_numpy(exp_eigenvec).float().cuda() / 10000

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # torch image: C X H X W
        img = transform.resize(image, (new_h, new_w))
        img = img.transpose((2, 0, 1))

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # swap color axis because
        # numpy image: H x W x C
        return torch.from_numpy(data).float()


#Batch LOader
BatchLoader = DataLoader(LFWLoader(),batch_size=4,shuffle=True,num_workers=4)

# UNIT TESTING
if __name__ == '__main__':

    import cv2
    import util

    # test testing loader
    loader = W300Loader()
    sample = loader[1]
    img = sample['image']
    lm2d = sample['lm2d']
    print(img.shape)
    print(lm2d.shape)
    img = img.permute(1,2,0)
    lm2d = lm2d.data.numpy() * 112
    lm2d = lm2d + 112
    myimg = img.data.numpy() * 255

    io.imsave('example.png',myimg)
    util.viewLM(myimg,lm2d)
    io.imsave('annotated.png',myimg)
    quit()

    # test training loader
    loader = LFWLoader()
    sample = loader[1]
    img = sample['image']
    lm2d = sample['lm2d']

    print("image shape",img.shape)
    print("landmark shape",lm2d.shape)

