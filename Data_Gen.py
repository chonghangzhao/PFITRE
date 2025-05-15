import glob
import os
import numpy as np
import tifffile as tf
from skimage.io import imsave, imread
from skimage import color, transform, exposure
import math
import tomopy
import astra
import random
from skimage import img_as_float, img_as_ubyte
from tqdm import tqdm
import scipy
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import rotate as rot
from numpy.random import choice

from scipy.sparse.linalg import gmres, lgmres, LinearOperator
from skimage.util.shape import view_as_windows as viewW

import time
import pandas as pd
import torch.nn as nn
import skimage.io as io

from PIL import Image

import matplotlib.pyplot as plt

from utils import *
from numpy import linalg as LA
from skimage.transform import rescale, downscale_local_mean


def file_to_process(image_inpath, Full_folder=True):
    """
    image_inpath: str
        The path for loading images to process
        
    Full_folder:
        True: load all images in designated folder 
        number: int, number of image to be randomly selected
    
    Return
        Filenames for processing 
    """

    random.seed(12345)
    
    num_jpg = len(sorted(glob.glob(image_inpath + '*.jpg')))
    num_tiff = len(sorted(glob.glob(image_inpath + '*.tiff')))
    if num_jpg==0 and num_tiff>1:
        image_infile0 = sorted(glob.glob(image_inpath + '*.tiff'))
    elif num_tiff==0 and num_jpg>1:
        image_infile0 = sorted(glob.glob(image_inpath + '*.jpg'))

    if Full_folder is True:
        image_infile = image_infile0
    elif isinstance(int(Full_folder), int) is True:
        a = random.sample(range(len(image_infile0)),int(Full_folder)) 
        image_infile = [image_infile0[j] for j in a]
    
    return image_infile


def imageto2d(image0):
    """
    image0:
       2D or 3D image 
    
    Return
        2D image

    convert RGB image in original training dataset to 2D 
    
    """
    if len(image0.shape) == 3:
        image0 = np.squeeze(image0)
        image0 = np.average(image0,axis=2) 
        
    elif len(image0.shape) == 2:
        image0 = np.squeeze(image0)
        
    return image0

def centerCrop_rotsimp(image, rot_angle=0, width_target=None, length_target=None):
    """
    image:
        2D image to be cropped
    
    rn: int
        rotation angle 
    
    width_target, length_target: int
        the expected dimension of image
    
    Return
        rotated image

    This function is used for data augmentation. 
    It will rotate the original image to specific angle, and resize to expected size.
    
    """
    width_org, length_org = image.shape

    if width_target == None and length_target == None:
         width = width_org
         length = length_org

    image = rot(image, angle = rot_angle, reshape=False)
    width_rot, length_rot = image.shape
    ### crop the image
    image2 = image[round(width_rot/2-width_org/2.828):round(width_rot/2+width_org/2.828), round(length_rot/2-length_org/2.828):round(length_rot/2+length_org/2.828)]
    ## resize to designated size
    image2 = transform.resize(image2, (width,length))
    return image2


def intensity_flipV2(image):
    """
    image: array
        2D image to be processed

    Return
        2D image after intensity rescaled
    
    This function is used for change the intensity of image. 
    In general X-ray images, their intensity at boundary is close to 0, and the intensity of central is close to 1. 
    It will calculate the intensity of boundary and central region. 
    If the intensity of central region is lower than boundary's, it will invert the intensity of image
    
    """
    
    ## intensity of boundary region
    bound_width = math.floor(min(image.shape)*0.05)
    bound1 = np.mean(image[0:bound_width, :])
    bound2 = np.mean(image[image.shape[0]-bound_width:image.shape[0], :])
    bound3 = np.mean(image[:, 0:bound_width])
    bound4 = np.mean(image[:, image.shape[1]-bound_width:image.shape[1]])

    bound_mean = np.mean([bound1, bound2, bound3, bound4])

    center_width = bound_width*8
    
    ## intensity of central region
    center0 = np.mean(image[int(image.shape[0]/2-bound_width*3):int(image.shape[0]/2+bound_width*3), int(image.shape[1]/2-bound_width*3):int(image.shape[1]/2+bound_width*3)])
    ## intenstiy of cross region 
    center1 = np.mean(image[int(image.shape[0]/2-center_width/2):int(image.shape[0]/2+center_width/2), int(image.shape[1]/2-bound_width):int(image.shape[1]/2+bound_width)])
    center2 = np.mean(image[int(image.shape[0]/2-bound_width):int(image.shape[0]/2+bound_width), int(image.shape[1]/2-center_width/2):int(image.shape[1]/2+center_width/2)])

    center_mean = np.mean([center0, center0, center1, center2])

    if center_mean<bound_mean:
        image = abs(image-bound_mean)
    else:
        image = image
        
    return image


def noisy_sino(sinogram, noise_type, noise_level_Gaussian, noise_level_Poisson):
    
    """
    noise_type:
        'Poission'
             add Poisson noise
        'Gaussian'
             add Gaussian noise
        'Mixed'
             add Gaussian and Poisson noise
    
    noise_level_Gaussian:
        'Random'
            apply random intensity of noise on input sinogram 
        float
            introduce Gaussian noise
    
    noise_level_Poisson:
        'Random'
            apply random intensity of noise on input sinogram 
        float
            muliply the intensity of image, and introduce different level of Poisson noise    
            
    """
    
    if noise_type == 'Poisson' or noise_type == 'Mixed':
         ### generate poisson noise
            
        #### shift the intesity of sinogram to positive value
        inten_shift = sinogram.min()
        if inten_shift<0:
            sinogram = sinogram+inten_shift          
        ## scale image to max is 1
        max_intense = sinogram.max()
        sinogram = sinogram/max_intense
        
        if isinstance(noise_level_Poisson, int)==True: 
        ## rescale to higher intenstiy, the noise level is actually scale factor
            scale_factor = noise_level
            sinogram = sinogram*noise_level ##scale_factor 
        elif noise_level_Poisson=='Random':
            scale_factor = np.random.normal(600, 80)  ##200-80, >=100  ##4200, 1800, >=1000 ###0606 for 300~80, >=200 ## middle for 1500~580, >=400 ## for higher 150-80 >=60
            if scale_factor < 60:  #100   ##15080_60  ##10080_40
                scale_factor = 60  ###
            sinogram = sinogram * scale_factor
        noise_poisson = np.random.poisson(lam=sinogram, size = None)
        sinogram = sinogram+noise_poisson
        sinogram = sinogram/scale_factor  ##scale_factor 

        sinogram = sinogram*max_intense
        if inten_shift<0:
            sinogram = sinogram-inten_shift     

    if noise_type == 'Gaussian' or noise_type == 'Mixed':    

        ### generate gaussian noise
        sino_L, sino_H = sinogram.shape
        if isinstance(noise_level_Gaussian, float)==True:
            noise_gaussian=np.random.normal(0, noise_level, [sino_L, sino_H])
        elif noise_level_Gaussian=='Random':
            noise_sigma = abs(np.random.normal(0, 0.6, 1))  ##for gaussian was 2.6 ##0.26 ## was 1200, 200, 1## 0~0.8 for mid noise ## 0-2 for high noise
            noise_gaussian = np.random.normal(0, noise_sigma, [sino_L, sino_H])

        sinogram = sinogram + noise_gaussian
                    
    elif noise_type == None:
        sinogram = sinogram
        
    return sinogram


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a,a[:,:-1]),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext,(1,n))[np.arange(len(r)), (n-r)%n,0]

def swap_elements(x, t):
    new_x = x[:]
    for idx, value in zip(choice(range(len(x)), size=len(t), replace=False), t):
        new_x[idx] = value
    return new_x

def sino_shiftX(sinogram):
    """
    simulate alignment issue in sinogram
    
    
    shift sinogram at each angle along X axis for random pixels, the shift range is from -2 to 2
    
    In 2~5 projection, it will randomly shift for -6~6 pixels

    """
    lower_limit = np.random.randint(-2,0)   # was +-4
    higher_limit = np.random.randint(1,2)

        
    shiftl = np.random.randint(lower_limit,higher_limit,sinogram.shape[0])
    
    extreme_shift_num = abs(np.random.randint(2,5))
    b = np.random.randint(-6,6,extreme_shift_num)
    shiftl = swap_elements(shiftl, b)   
    
    sino2 = strided_indexing_roll(sinogram, shiftl)
    
    return sino2

def out_FOV(sinogram):
    """
    simulate out of Field of View issue
    
    
    """
    miss_idx_list = []
    for idx0 in range(sinogram.shape[0]):
        miss_idx_list.append(random.randrange(0,15))
    miss_idx_list = [a if a<10 else 0 for i, a in enumerate(miss_idx_list)]

    sino_zeros = np.zeros(sinogram.shape)
    for i in range(len(miss_idx_list)):
        sino_zeros[i,miss_idx_list[i]:-1] = sinogram[i,miss_idx_list[i]:-1]
    sinogram = sino_zeros
    
    return sinogram

def sino_normY(sinogram):
    """
    randomly adjust intensity variation on the sinogram to simulate normalization issue
    """
    
    multipliers = np.random.uniform(0.95, 1.05, sinogram.shape[0])  # was 0.9~1.1
    sinogram = sinogram*multipliers[:, np.newaxis]
    
    return sinogram


def img_individual_norm_3Dformat(source_work_dir, out_dir, gt):
    
    """
    source_work_dir: 
        the input dataset needs to be normalized
    
    out_dir: 
        output path
    
    gt:
        True
            if the image is ground truth
        False
            if the image is with artifact 
    
    the original ground truth image is in 2D format, to fit into VGG loss, it will be reformat to 3D.
    image with artifact is 3D format
    
    This functio will firstly scale image intensity to 0-1,
    and then normalized each image with its mean and std.
    
    """
    
    img_fl = sorted(glob.glob(source_work_dir+'/*.tiff'))  ##/Validated

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for i in tqdm(range(len(img_fl))):
        img_int0=tf.imread(img_fl[i])
        fn = os.path.basename(img_fl[i])
        
        ## scale the image to 0-1
        img_int0 = (img_int0-img_int0.min())/(img_int0.max()-img_int0.min()) 
        
        ## normalize by mean and std
        imgm0 = img_int0.mean()
        imgstd0 = img_int0.std()
        img_norm = (img_int0-imgm0)/imgstd0

        ## change format to 3D if it is a ground truth image
        if gt==True:
            img_norm = np.expand_dims(img_norm, axis=0) 
            img_norm = np.concatenate((img_norm,img_norm,img_norm))
        else:
            img_norm = img_norm[:,:,0]
            img_norm = np.expand_dims(img_norm, axis=0)
            img_norm = np.concatenate((img_norm,img_norm,img_norm))

        out_fn = os.path.join(out_dir, fn)
        tf.imsave(out_fn, img_norm)


def Data_augment(image, resize=False, target_size=320, rotate=False, rot_angle=0):
    """
    image:
        ndarray, 2d image or 3d RGB image will be reduced to 2D
        
    resize:
        True, False: whether resize images or not
        
    target_size:
        int, target size in dataset
    
    rotate:
        True, False: whether rotate the image for augmentation or not
        
    rot_angle:
        int: rotation angle
    
    Return
        Augmented 2D image, after scaled to 0-1 range
    """
    
    image2d = imageto2d(image)
    
    if resize is True:
        image2d = transform.resize(image2d, (int(target_size), int(target_size)))  
    
    if rotate is True and resize is True:
        image2d = centerCrop_rotsimp(image2d, rot_angle, target_size, target_size)
    elif rotate is True and resize is False:
        image2d = centerCrop_rotsimp(image, rot_angle, target_size, target_size)
        
    ## rescale image to all positive values
    if image2d.min()<0:
        image2d = image2d-image2d.min()

    ## scale image intensity to 0-1
    image2d = (image2d-image2d.min())/(image2d.max()-image2d.min())
    ## check the intensity, if boundary intensity is higher thenthe ceter region, then flip the intensity so that background becomes dark
    image2d = intensity_flipV2(image2d)
            
    return image2d

def angle_list_gen(miss_angle="Random", step=1, rand_int=0, rot_angle=0, fn='NotSpecified'):
    """
    create angle list for simulating missing angle artifact
    
    miss_angle:
        int: specify the missing angle 
        'Random': missing angle randomly select from 30-90
        
    step:
        float: angular step size 
    
    rand_int:
        int: include random integer angle as the offset 
        
    fn:
        str: for incorporate rotation information
    
    
    Returns
    -------
    array
        projection angles in radian

    """
    if miss_angle == "Random":
        miss_ang = int(np.random.randint(30,91,1))  #was 20-121
    else:
        miss_ang = int(miss_angle)
    
    ### for asymmetric projection
    angle = 180-miss_ang
    

    ### this is only for new chip dataset
    ## missing along the horizontal direction
    if 'rot24' in os.path.basename(fn):
        init = 114+12
    elif 'rot114' in os.path.basename(fn):
        init = 24+12
    else: 
        init = 0 #random.randint(-20,20)
    random_start = init + int(rot_angle) +int(rand_int) #random.randint(-20,20) ##
    
    ## angle range to generate image
    theta2 = np.linspace(float(0-angle/2+random_start), float(angle/2+random_start), int(angle/step), endpoint=False)   
    theta = np.array(theta2)/180.*np.pi
    
    return theta


def Gen_sino(image, theta, cor_shift=0, padding=True):
    """
    image: ndarray
        2D gt image to generate sinogram with artifact
        
    fn: str
        filename of the image, 
        It is useful for generating artifact image for line shape feature, where line shape orientation needs to know
        
    cor_shift: float
        rotation center offset
        
    step: float
        angular step size 
    
    padding:
        True
            pad the image and generate sinogram FOV
        False
            do not pad the square shape image, corner information will lost 
            
            
    Returns
    -------
    2darray
        sinogram
    ndarray
        covered angular range
    
    This reconstruction is developed upon ASTRA toolbox and tomobar. More detail can be found in below link     
    https://github.com/dkazanc/ToMoBAR/blob/master/src/Python/tomobar/supp/astraOP.py
    
    """

    width, length = image.shape
    # create projection by Astra toolbox
    pad_size = max(width, length)
    vol_geom = astra.create_vol_geom(width, length)
    
    if padding==True:
        proj_geom = astra.create_proj_geom('parallel', 1.0, pad_size*2, theta)
    else:
        proj_geom = astra.create_proj_geom('parallel', 1.0, pad_size, theta)
        
    P = np.squeeze(image) 
 
    ######
    if cor_shift != 0:
        ### include rotation center offset 
        proj_geom_cor = astra.geom_postalignment(proj_geom, cor_shift)
        proj_id = astra.create_projector('cuda',proj_geom_cor,vol_geom)
    elif cor_shift == 0:
        proj_id = astra.create_projector('cuda',proj_geom,vol_geom) 

#     ######
#     proj_id = astra.create_projector('cuda',proj_geom,vol_geom) 
    sinogram_id, sinogram = astra.create_sino(P, proj_id)
    
    astra.projector.delete(proj_id)

    return sinogram


def recon_by_solver(sinogram, theta, padding=True, cor_shift=0):
    """
    sinogram: ndarray
        2D gt image to generate image with artifact
        
    theta: array
        angles in radian
    
    padding:
        True
            pad the image and generate sinogram FOV
        False
            do not pad the square shape image, corner information will lost 
        
    cor_shift:
        rotation center offset with respect to the center of sinogram
    
    Returns
    -------
    2darray
        Reconstructed 2D object
        
    
    This reconstruction is developed upon ASTRA toolbox and tomobar. More detail can be found in below link     
    https://github.com/dkazanc/ToMoBAR/blob/master/src/Python/tomobar/supp/astraOP.py
    
    """
    width = sinogram.shape[1] 
      
    length = width 
    proj_geom = astra.create_proj_geom('parallel', 1.0, width, theta)
    
    if padding==True:
        vol_geom = astra.create_vol_geom(int(width/2), int(length/2))
        pad_size = int(width/2)
    elif padding==False:
        vol_geom = astra.create_vol_geom(width, length)
        pad_size = width
        
        
    ### rotation center shift    
    if cor_shift != 0:
        ### include rotation center offset 
        proj_geom_cor = astra.geom_postalignment(proj_geom, cor_shift)
        proj_id = astra.create_projector('cuda',proj_geom_cor,vol_geom)
    elif cor_shift == 0:
        proj_id = astra.create_projector('cuda',proj_geom,vol_geom) 
        

    ### reconstruction with linear solver in fidelity term 
    A_optomo = astra.OpTomo(proj_id)
    def ADMM_Ax(x):
        data_upd = A_optomo(x)
        x_temp = A_optomo.transposeOpTomo(data_upd)
        x_upd = x_temp# + ADMM_rho_const*x
        return x_upd
    def ADMM_Atb(b):
        b = A_optomo.transposeOpTomo(b)
        return b

    #### recon with tomobar methods
    ObjSize = pad_size
    (data_dim,rec_dim) = np.shape(A_optomo)
    X = np.zeros(rec_dim, 'float32') ## original was float 32!!
    b_to_solver_const = A_optomo.transposeOpTomo(sinogram.ravel())

    A_to_solver = scipy.sparse.linalg.LinearOperator((rec_dim,rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb)
    b_to_solver = b_to_solver_const
    outputSolver = scipy.sparse.linalg.gmres(A_to_solver, b_to_solver, tol = 1e-05, maxiter = 15)
    X = np.float32(outputSolver[0])
    
    recon = X.reshape([ObjSize, ObjSize])
    # normalize image to 0-1 range, remove artifacts from negative intensity 
    if (recon.max()==0) and (recon.min()==0):
        new_recon = recon
    else:
        new_recon = (recon-recon.min())/(recon.max()-recon.min())
    
    astra.projector.delete(proj_id)
    
    return new_recon


def recon_ADMM_NN(sinogram, theta, Model, ckpt_dir, ckpt_num, iter_num=8, ADMM_rho_const=5, cor_shift=0, padding=True, initial=None):
    """
    sinogram: ndarray
        2D gt image to generate image with artifact
        
    theta: array
        angles in radian
        
    iter_num:
        int: number of iteration in ADMM 
    
    Model:
        the network to be used in ADMM as regularizer
        
    ckpt_dir:
        str: the path to where weight was saved 
    
    ckpt_num:
        int: the check point of network     
        
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
    
    Returns
    -------
    2darray
        Reconstructed 2D object

        
    
    This reconstruction is developed upon ASTRA toolbox and tomobar. More detail can be found in below link     
    https://github.com/dkazanc/ToMoBAR/blob/master/src/Python/tomobar/supp/astraOP.py
    
    """
    
    x_updatelist = []

    width = sinogram.shape[1] 

    ## this is for 2d projection
    length = width 

    ###### assume the input sinogram has been padded
    proj_geom = astra.create_proj_geom('parallel', 1.0, width, theta)
    
    if padding==True:
        vol_geom = astra.create_vol_geom(int(width/2), int(length/2))
        pad_size = int(width/2)
    elif padding==False:
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

    ######### pre-trained model as regularization in ADMM
    
    ckpt_dir = ckpt_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(ckpt_dir, "ckpt_"+str(ckpt_num)+".pth") 
    checkpoint = torch.load(model_path)

    UnetModel = Model        
    UnetModel.load_state_dict(checkpoint['net_state_dict'])
    UnetModel = UnetModel.to(device)
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
        z = initial#[selected_slice,:,:] 
    u = np.zeros(rec_dim, 'float32') 
    denomN = 1.0/np.size(X)
    
    nrm_list= []
    for iters in range(0,iter_num):
        X_old = np.ravel(X) 
        # solving quadratic problem using linalg solver
        A_to_solver = scipy.sparse.linalg.LinearOperator((rec_dim,rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb)
        b_to_solver = b_to_solver_const + ADMM_rho_const*(z-u)
        outputSolver = scipy.sparse.linalg.gmres(A_to_solver, b_to_solver, x0=np.ravel(z), tol = 1e-05, maxiter = 15) 
        X = np.float32(outputSolver[0]) 
        
        X[X < 0.0] = 0.0
        
        # z-update with relaxation
        zold = z.copy();
        x_hat = ADMM_relax_par*X + (1.0 - ADMM_relax_par)*zold;
        if (geom == '2D'):
            x_prox_reg = (x_hat + u).reshape([ObjSize, ObjSize])
        if (geom == '3D'):
            x_prox_reg = (x_hat + u).reshape([DetectorsDimV, ObjSize, ObjSize])
            
        x_updatelist.append(x_prox_reg)  

        x_prox_reg[x_prox_reg < 0.0] = 0.0

        ### scale recon image to 0-1 first
        ### normalize the input image based on its mean and std 
        x_prox_reg2d, x_prox_reg_min, x_prox_reg_max = image2d_tomodel_range(x_prox_reg, togpu=False)
        x_prox_reg2d = norm_inNout(x_prox_reg2d, UnetModel, norm='self')
        z = x_prox_reg2d*(x_prox_reg_max-x_prox_reg_min)+x_prox_reg_min

        ### regularize network output
        z[z < 0.0] = 0.0

        z = z.ravel()
        # update u variable
        u = u + (x_hat - z)

    astra.projector.delete(proj_id)
    
    return X.reshape([ObjSize, ObjSize]), x_updatelist



def img_2d_to_3d(recon):
    """
    image from 2D to 3D format
    
    recon
    2D array image
    
    Returns:
    ---------
    3d array
    
    """
    
    recon3d = np.expand_dims(recon,axis=0)
    recon3d = np.concatenate((recon3d,recon3d,recon3d),axis=0)
    reconT = np.transpose(recon3d,(1,2,0))

    return reconT
