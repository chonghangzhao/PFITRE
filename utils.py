import os
import glob
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
import tifffile as tf
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import rotate
from scipy.ndimage import rotate as rot
from numpy.random import choice

import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import scipy
from scipy.sparse.linalg import gmres, lgmres, LinearOperator
from tqdm import tqdm
import time

import math
import tomopy
import astra

from skimage.util.shape import view_as_windows as viewW
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_bregman, denoise_tv_chambolle, denoise_bilateral
import skimage.io as io
from PIL import Image
from numpy import linalg as LA
from skimage.transform import rescale, downscale_local_mean


random.seed(12345)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image2d_tomodel_range(test, togpu=True):
    '''
    rescale the intensity of image to 0-1, based on absolute min and max value in the image
    togpu:
        True: 2d image will be converted to 3d tensor,
        False: 2d image remains
    '''
    t_min = test.min()
    t_max = test.max()
    test= (test-test.min())/(test.max()-test.min())

    if togpu is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test = np.expand_dims(test, axis=0) 
        test = np.concatenate((test,test,test))
        test = np.expand_dims(test, axis=0)
        test = torch.tensor(test,dtype=torch.float32, requires_grad=False)
        test = test.to(device)
    elif togpu is False:
        test = test

    return test, t_min, t_max


def image2d_tomodel_range_quantile(test, percentile=0.01, togpu=True):
    '''
    rescale the intensity of image to 0-1, based on percentile of pixel intensity in the image.
    togpu:
        True: 2d image will be converted to 3d tensor,
        False: 2d image remains
    '''

    t_min = np.quantile(test, q=percentile)
    t_max = np.quantile(test, q=(1-percentile))
    test= (test-test.min())/(test.max()-test.min())

    if togpu is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test = np.expand_dims(test, axis=0) 
        test = np.concatenate((test,test,test))
        test = np.expand_dims(test, axis=0)
        test = torch.tensor(test,dtype=torch.float32, requires_grad=False)
        test = test.to(device)
    elif togpu is False:
        test = test

    return test, t_min, t_max

def img_pad(img_tensor, pad_mod='circular'):
    '''
    Pad the size of the square image to a multiple of 4

    pad_mod(str):
        'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        please refer to torch.nn.functional.pad

    '''
    pad_len_horiz = 4-(img_tensor.size()[-1])%4
    pad_len_vert = 4-(img_tensor.size()[-2])%4
    
    if (pad_len_horiz ==4):
        padding_dimen = (0,0,0,0)
    elif (pad_len_horiz%2 ==0):
        padding_dimen = (int(pad_len_horiz/2), int(pad_len_horiz/2), 0, 0)
    elif (pad_len_horiz%2 !=0):
        padding_dimen = (int(math.ceil(pad_len_horiz/2)),int(math.ceil(pad_len_horiz/2)-1), 0, 0)
    
    print(padding_dimen)
    padded_width_tensor = F.pad(img_tensor, padding_dimen, mode=pad_mod)

    # Now, pad top and bottom (height)
    
    if (pad_len_vert ==4):
        padding_dimen2 = (0,0,0,0)
    elif (pad_len_vert%2 ==0):
        padding_dimen2 = (0, 0, int(pad_len_vert/2), int(pad_len_vert/2))
    elif (pad_len_vert%2 !=0):
        padding_dimen2 = (0, 0, int(math.ceil(pad_len_vert/2)),int(math.ceil(pad_len_vert/2)-1))
    padded_tensor = F.pad(padded_width_tensor, padding_dimen2, mode=pad_mod)

    print(padding_dimen2)

    return padded_tensor


def image_tomodel(test):
    '''
    convert RGB image to 3d tensor
    '''
    test = np.average(test, axis=-1)#[:,:,0]
    test = np.expand_dims(test, axis=0)
    test = np.concatenate((test,test,test))

    test = np.expand_dims(test, axis=0)
    test = torch.tensor(test,dtype=torch.float32, requires_grad=False)
    test = test.to(device)
    return test


def image2d_tomodel(test):
    '''
    convert 2d image to 3d tensor
    '''

    test = np.expand_dims(test, axis=0) 
    test = np.concatenate((test,test,test))

    test = np.expand_dims(test, axis=0)
    test = torch.tensor(test,dtype=torch.float32, requires_grad=False)
    test = test.to(device)
    return test

def output_tocpu(model_output,plot2D=False):
    a = model_output.cpu().detach().numpy()
    a = np.squeeze(a)
    if plot2D == True:
        a = np.average(a, axis=0)
    return a

def norm_inNout(img_in, Model, norm='self', pad_mod='constant', percentile=0.01):
    '''
    for normalizing image based on its mean and std from input or training statistics
    
    norm(str):
        self:rescale and normalize the image, based on absolute min and max value in the image
        self_quant: rescale and normalize the image, based on percentile of pixel intensity in the image

    pad_mod(str):
        To pad image size, and fit into model
        'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        please refer to torch.nn.functional.pad
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(img_in.shape) == 3:
        art_object = image_tomodel(img_in)
    elif len(img_in.shape) == 2:
        art_object = image2d_tomodel(img_in)
    else:
        print("Input format can only be 2d or 3d np array")

    if pad_mod is None:
        art_object = art_object
    else:
        art_object = img_pad(art_object, pad_mod=pad_mod)
    
    if norm == 'self':
        ## if use the data itself's mean and std for normalization
        mean_art = art_object.mean()
        mean_art = torch.tensor((mean_art,mean_art,mean_art)).to(device)
        std_art = art_object.std()
        std_art = torch.tensor((std_art,std_art,std_art)).to(device)

    elif norm == 'self_quant':
        ## if use the data itself's mean and std for normalization
        ### calculate mean and std based on filtered dataset, remove potential hot pixels issue, out of the target quantile
        low_quant_value = torch.quantile(art_object, percentile)
        high_quant_value = torch.quantile(art_object, (1-percentile))
        art_object_filter = art_object[(art_object>=low_quant_value)&(art_object<=high_quant_value)]
        
        mean_art = art_object_filter.mean()
        mean_art = torch.tensor((mean_art,mean_art,mean_art)).to(device)
        std_art = art_object_filter.std()
        std_art = torch.tensor((std_art,std_art,std_art)).to(device)

    art_object2 = (art_object-mean_art[:, None, None] ) / std_art[:, None, None]
    Model.to(device)
    img_out = Model(art_object2)
    img_out_norm = img_out*std_art[:, None, None] + mean_art[:, None, None]
    img_out_norm = output_tocpu(img_out_norm, plot2D=True)

    return img_out_norm

def NN_load(Model, ckpt_path):
    """
    Model:
        the neural network architecture to be used
        
    ckpt_path:
        Path to the model weight 
        
    ----------------
    Return:
        network with pretrained weight 
        
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    model_path = os.path.join(ckpt_dir, "ckpt_"+str(int(ckpt_num))+".pth")
    checkpoint = torch.load(ckpt_path)

    Model.load_state_dict(checkpoint['net_state_dict'])
    Model = Model.to(device)
    
    return Model 


def NN_PostCorrect(img_input, Model, norm='self', norm_quant=0.01, pad_mod='constant'):
    """
    This function only works for one time post processing on 2d single image. 

    img_input:
        2d or 3d array
        
    Model:
        Neural network to be used
        
    norm:
        The way of doing normalization. The default is 'self'.
        "self": normalize image to 0-1, based on absolute max and min value in the input image
        "self_quant": normalize image to roughly 0-1, based on percentile value in the input image. 
		      This is design for handling images with hot spot issue. 
    norm_quant:
        It only take effect when choosing 'self_quant' for image rescale. 
        The quantile value selected here is used for representing min and max value for rescale. 
        The default value is 0.01, which means min and max value for rescale is calculated based on 0.01 and 0.99 quantile in the input images.

    pad_mod(str):
        To pad image size, and fit into model
        'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        please refer to torch.nn.functional.pad
    ----------------
    
    Return:
        2d np array

    """
    if norm == 'self':
        ## rescale based on absolute max and min value in a image

        img_2d, img_min, img_max = image2d_tomodel_range(img_input, togpu=False)
        img_2d = norm_inNout(img_2d, Model, norm='self', pad_mod=pad_mod)  
        model_output = img_2d*(img_max-img_min)+img_min
        
    elif norm == 'self_quant': 
        ## rescale based on percentile value in a image
        img_2d2, img_min, img_max = image2d_tomodel_range_quantile(img_input, percentile=norm_quant, togpu=False)
        img_2d2 = norm_inNout(img_2d2, Model, norm='self_quant', percentile=norm_quant, pad_mod=pad_mod)
        model_output = img_2d2*(img_max-img_min)+img_min
        
    return model_output



def recon_ADMM_NN_TV(sinogram, theta, Model, iter_num=8, ADMM_rho_const=5, cor_shift=0, padding=True, initial=None, mask_boundary=False, mask_ratio=0.95, norm_quant=False, TV=True):
    """
    sinogram: ndarray
        2D gt image to generate image with artifact
        
    theta: array
        angles in radian
        
    iter_num:
        int: number of iteration in ADMM 
    
    Model:
        the network to be used in ADMM as regularizer
        
    ADMM_rho_const
        int: the weight of regularization term, the higher the more contribution from regularization
        
    cor_shift:
        rotation center offset with respect to the center of sinogram
            
    padding:
        True
            pad the image and generate sinogram FOV, this needs to be consistent with sinogram generation function
        False
            do not pad the square shape image, corner information will lost 
            
    initial:
        2d array: initial guess for ADMM 
        
    mask_boundary:
        True: mask the boundary as zero
        False: do not mask and directly load image to Network
    
    mask_ratio:
        float: ratio of diameter of mask to the image width, default is 0.95
        This is used for removing high intense ring artifact at the boundary
    
    norm_quant:
        float: percentile as max and min for image normalization process
        
    Returns
    -------
    2darray
        Reconstructed 2D object

        
    
    This reconstruction is developed upon ASTRA toolbox and tomobar. More detail can be found in below link     
    https://github.com/dkazanc/ToMoBAR/blob/master/src/Python/tomobar/supp/astraOP.py
    
    """
    
    x_updatelist = []

    pad_len = 4-sinogram.shape[1]%4

    if pad_len%2 ==0:
        sinogram = np.pad(sinogram, ((0,0),(int(pad_len/2),int(pad_len/2))),mode='constant', constant_values=(0,0))  #,mode='minimum') 
    else:
        sinogram = np.pad(sinogram, ((0,0),(int(np.round(pad_len/2)),int(np.round(pad_len/2)+1))),mode='constant', constant_values=(0,0))  #, mode='minimum') 

    width = sinogram.shape[1] 
    ## this is for 2d projection
    length = width 
    
    proj_geom = astra.create_proj_geom('parallel', 1.0, width, theta)
    
    vol_geom = astra.create_vol_geom(width, length)
    pad_size = width
    
    ### rotation center shift    
    if cor_shift != 0:
        ### include rotation center offset 
        proj_geom_cor = astra.geom_postalignment(proj_geom, cor_shift)
        proj_id = astra.create_projector('cuda',proj_geom_cor,vol_geom)
    elif cor_shift == 0:
        proj_id = astra.create_projector('cuda',proj_geom,vol_geom) 
        
    ###############################################
    A_optomo = astra.OpTomo(proj_id)
    ADMM_rho_const = ADMM_rho_const ##5
    ADMM_relax_par = 1.0
    
    nonnegativity = "DISABLE"
    ObjSize = pad_size
    geom = str(len(sinogram.shape))+"D"
    tolerance = 0

    ####################
    
    def ADMM_Ax(x):
        data_upd = A_optomo(x)
        x_temp = A_optomo.transposeOpTomo(data_upd)
        x_upd = x_temp + ADMM_rho_const*x
        return x_upd
    def ADMM_Atb(b):
        b = A_optomo.transposeOpTomo(b)
        return b
    
    ObjSize = pad_size
    (data_dim,rec_dim) = np.shape(A_optomo)
    X = np.zeros(rec_dim, 'float32')  
    x_prox_reg = np.zeros(rec_dim, 'float32')
    b_to_solver_const = A_optomo.transposeOpTomo(sinogram.ravel())
    if initial==None:
        z = np.zeros(rec_dim, 'float32')  
    else:
        z = initial
    u = np.zeros(rec_dim, 'float32') 
    denomN = 1.0/np.size(X)
    
    nrm_list= []
    for iters in tqdm(range(0,iter_num)):
        X_old = np.ravel(X) 
        # solving quadratic problem using linalg solver
        A_to_solver = scipy.sparse.linalg.LinearOperator((rec_dim,rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb)
        b_to_solver = b_to_solver_const + ADMM_rho_const*(z-u)
        outputSolver = scipy.sparse.linalg.gmres(A_to_solver, b_to_solver, x0=np.ravel(z), tol = 1e-05, maxiter = 15) 
        X = np.float32(outputSolver[0]) 
        
        X[X < 0.0] = 0.0
        
        x_updatelist.append(X.reshape([ObjSize, ObjSize]))  
        # z-update with relaxation
        zold = z.copy();
        x_hat = ADMM_relax_par*X + (1.0 - ADMM_relax_par)*zold;
        x_prox_reg = (x_hat + u).reshape([ObjSize, ObjSize])

        x_prox_reg[x_prox_reg < 0.0] = 0.0
        
        
        ################
        if mask_boundary is True:
            h,w = x_prox_reg.shape
            radius = int(h/2*mask_ratio)
            mask = create_circular_mask(h, w, radius=radius)
            masked_img = x_prox_reg.copy()
            masked_img[~mask] = 0.0
            x_prox_reg = masked_img

        ### scale recon image to 0-1 first
        ### normalize the input image based on its mean and std 
        
        if isinstance(norm_quant, float)==True:
            x_prox_reg2d, x_prox_reg_min, x_prox_reg_max = image2d_tomodel_range_quantile(x_prox_reg, percentile=norm_quant, togpu=True)
            x_prox_reg2d = norm_inNout(x_prox_reg2d, Model, norm='self_quant', percentile=norm_quant, pad_mod=None)
        elif norm_quant is False:
            x_prox_reg2d, x_prox_reg_min, x_prox_reg_max = image2d_tomodel_range(x_prox_reg, togpu=False)
            x_prox_reg2d = norm_inNout(x_prox_reg2d, Model, norm='self', pad_mod=None)
        z = x_prox_reg2d*(x_prox_reg_max-x_prox_reg_min)+x_prox_reg_min

        ### regularize network output
        z[z < 0.0] = 0.0
        
        ### include TV regularization after NN processing

        if TV is True:
            z = denoise_tv_bregman(z, weight=200, eps=0.05)
        elif TV is False:
            z = z
        
        z = z.ravel()
        # update u variable
        u = u + (x_hat - z)

    astra.projector.delete(proj_id)
    
    return X.reshape([ObjSize, ObjSize]), x_updatelist


