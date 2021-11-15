# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:14:34 2019


@author: chenlin

test the learned aff modules on hpatches dataset
"""


import matplotlib
matplotlib.use('Agg')
import os
import errno
import numpy as np
from PIL import Image
import sys
from copy import deepcopy
import argparse
import math
import torch.utils.data as data
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import gc
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import cv2
import copy
from Utils import L2Norm, cv2_scale
#from Utils import np_reshape64 as np_reshape
np_reshape = lambda x: np.reshape(x, (64, 64, 1))
from Utils import str2bool
from dataset import HPatchesDM,TripletPhotoTour, TotalDatasetsLoader
cv2_scale40 = lambda x: cv2.resize(x, dsize=(40, 40),
                                 interpolation=cv2.INTER_LINEAR)
from augmentation import get_random_norm_affine_LAFs,get_random_rotation_LAFs, get_random_shifts_LAFs
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches,normalizeLAFs
from architectures import AffNetFast, AffNetFastScale, AffNetFast4, AffNetFast4RotNosc, \
AffNetFast52RotUp,AffNetFast52Rot,AffNetFast5Rot, AffNetFast4Rot, AffNetFast4Rot, OriNetFast
from architectures import AffNetFast2Par,AffNetFastBias
from pytorch_sift import SIFTNet
from HardNet import HardNet, L2Norm, HardNet_Chen
#from Losses import loss_HardNetDetach, loss_HardNet
from Losses import loss_HardNet
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A,visualize_LAFs
import seaborn as sns
from Losses import distance_matrix_vector
from ReprojectionStuff import get_GT_correspondence_indexes
import argparse

USE_CUDA = True

parser = argparse.ArgumentParser(description='PyTorch Affnet_HardnetChen')

parser.add_argument('--weightd_fname_affnet', type=str,
                    default='./logs/AFFNet_OriNetFast_Joint_lr00001_10M_20ep_aswap_ipidata_3108_AffNetFast_6Brown_HardNet_1e-05_10000000_HardNet/checkpoint_aff_17.pth',
                    help='path to affnet weights')
parser.add_argument('--weightd_fname_orinet', type=str,
                    default='./logs/AFFNet_OriNetFast_Joint_lr00001_10M_20ep_aswap_ipidata_3108_AffNetFast_6Brown_HardNet_1e-05_10000000_HardNet/checkpoint_ori_17.pth',
                    help='path to orientation part weight')
parser.add_argument('--descriptor_model_weights', type=str,
                    default="../data/models/all_min_30000000_as/checkpoint_19.pth",
                    help='path to descriptor weights')



args = parser.parse_args()

### Initialization
AffNetPix = AffNetFast(PS = 32)
#weightd_fname = './pretrained/AffNet.pth'
#weightd_fname = 'logs/AffNetFast_lr005_10M_20ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNegC/checkpoint_18.pth'

#weightd_fname = 'logs/AffNetFast_hardest_k_lr005_10M_25_0715ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNet_k_hardest/checkpoint_24.pth'
#weightd_fname = 'logs/AffNetFast_hardest_k_lr005_10M_25_0715ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNet_k_hardest/checkpoint_7.pth'
#weightd_fname = 'logs/AffNetFast_hardest_k_4_lr0001_10M_25_0715ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.0001_10000000_HardNet_k_hardest/checkpoint_24.pth'
#weightd_fname = 'logs/AffNetFast_hardest_k_4_lr0001_10M_25_0715ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.0001_10000000_HardNet_k_hardest/checkpoint_24.pth'

#weightd_fname = 'logs/AffNetFast_hardest_k_lr005_10M_25ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNet_k_hardest/checkpoint_7.pth'
#weightd_fname = 'logs/AffNetFast_hardNet_HardnetDesc_lr0001_Adam_10M_15_2608ep_aswap_dortmund_AffNetFast_6Brown_HardNet_0.0001_10000000_HardNegC/checkpoint_14.pth'
#weightd_fname = 'logs/AffNetFast_hardNet_HardnetDesc_lr0001_Adam_10M_15_2108ep_aswap_dortmund_AffNetFast_6Brown_HardNet_0.0001_10000000_HardNet_k_hardest/checkpoint_12.pth'
#weightd_fname_affnet = './logs/AFFNet_OriNetFast_Joint_lr00001_10M_20ep_aswap_ipidata_3108_AffNetFast_6Brown_HardNet_1e-05_10000000_HardNet/checkpoint_aff_17.pth'
weightd_fname_affnet = args.weightd_fname_affnet
checkpoint = torch.load(weightd_fname_affnet)
AffNetPix.load_state_dict(checkpoint['state_dict'])

AffNetPix.eval()

#load orientation net for more training process

OriNetPix = OriNetFast(PS=32)

def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    if USE_CUDA:
        var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape
def get_geometry_and_descriptors(img, det, desc, do_ori = True):
    with torch.no_grad():
        LAFs, resp = det(img,do_ori = do_ori)
        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = desc(patches)
    return LAFs, descriptors

#def get_geometry_and_descriptors_response(img, det, desc):
#    with torch.no_grad():
#        LAFs, resp = det(img, do_ori = True)
#        patches = detector.extract_patches_from_pyr(LAFs, PS = 32)
#        descriptors = descriptor(patches)
#    return LAFs, descriptors, resp

#def get_geometry_and_descriptors(img, det, desc, do_ori = True):
#    with torch.no_grad():
#        LAFs, resp = det(img,do_ori = do_ori)
#        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
#        descriptors = desc(patches)
#    return LAFs, descriptors



from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid, NMS3dAndComposeA

def test_on_hpatchsequence(hpatches_sequence_path):
    torch.cuda.empty_cache()
    # switch to evaluate mode
    
    OriNetPix = OriNetFast(PS=32)
#    weightd_fname_orinet = './pretrained/OriNet.pth'
#    weightd_fname_orinet = './logs/AFFNet_OriNetFast_Joint_lr00001_10M_20ep_aswap_ipidata_3108_AffNetFast_6Brown_HardNet_1e-05_10000000_HardNet/checkpoint_ori_17.pth'
    weightd_fname_orinet = args.weightd_fname_orinet
#    weightd_fname_orinet = 'logs/OriNetFast_lr005_10M_20ep_aswap_ipidata_OriNet_6Brown_HardNet_0.005_10000000_HardNet/checkpoint_' + str(19) + '.pth'
#    AffNetFast_hardest_k_lr005_10M_25_0715ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNet_k_hardest/checkpoint_' + str(19) + '.pth'
    checkpoint_orinet = torch.load(weightd_fname_orinet)
    OriNetPix.load_state_dict(checkpoint_orinet['state_dict'])
    OriNetPix.eval()
    
    detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 5000,
                                          border = 5, num_Baum_iters = 1, 
                                          AffNet = AffNetPix, OriNet = OriNetPix)
#    detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
#                                      border = 5, num_Baum_iters = 1, 
#                                      AffNet = AffNetPix)
    
#    aff_handc = AffineShapeEstimator(patch_size=19)
#    ori_handc = OrientationDetector(patch_size = 19)
#    
#    detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 5000,
#                                      border = 5, num_Baum_iters = 1,
#                                      AffNet = aff_handc,
#                                      OriNet = ori_handc)
    
    descriptor = HardNet_Chen()
    model_weights = args.descriptor_model_weights
#    "../data/models/all_min_30000000_as/checkpoint_19.pth"
    hncheckpoint = torch.load(model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    
    
#    from pytorch_sift import SIFTNet
#    descriptor = SIFTNet(patch_size=32)
    
    
    descriptor.eval()
    if USE_CUDA:
        detector = detector.cuda()
        descriptor = descriptor.cuda()
    
    list_truematch_number_view = []
    list_truematch_number_light = []
    
    folders = os.listdir(hpatches_sequence_path)
    
    for dataset in folders:
        
        input_img_fname1 = os.path.join(hpatches_sequence_path, dataset,'1.ppm')
        input_img_fname2 = os.path.join(hpatches_sequence_path, dataset,'6.ppm')
        H_fname = os.path.join(hpatches_sequence_path,folders[0],'H_1_6')
        
        img1 = load_grayscale_var(input_img_fname1)
        img2 = load_grayscale_var(input_img_fname2)
        H = np.loadtxt(H_fname)    
        H1to2 = Variable(torch.from_numpy(H).float())
        SNN_threshold = 0.8
        
        with torch.no_grad():
            LAFs1, descriptors1 = get_geometry_and_descriptors(img1, detector, descriptor)
            torch.cuda.empty_cache()
            LAFs2, descriptors2 = get_geometry_and_descriptors(img2, detector, descriptor)
            dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
            min_dist, idxs_in_2 = torch.min(dist_matrix,1)
            dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
            min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
            mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
            tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False).cuda()[mask]
    #        tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False)[mask]
            tent_matches_in_2 = idxs_in_2[mask]
            tent_matches_in_1 = tent_matches_in_1.long()
            tent_matches_in_2 = tent_matches_in_2.long()
            LAF1s_tent = LAFs1[tent_matches_in_1,:,:]
            LAF2s_tent = LAFs2[tent_matches_in_2,:,:]
            min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent,H1to2.cuda(), dist_threshold = 3.0) 
    #        min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent,H1to2, dist_threshold = 6) 
            plain_indxs_in1 = plain_indxs_in1.long()
            inl_ratio = float(plain_indxs_in1.size(0)) / float(tent_matches_in_1.size(0))
            print('Test dataset: ', dataset) 
            print('Test on img 1-6,', tent_matches_in_1.size(0), 'tentatives', plain_indxs_in1.size(0), 'true matches', str(inl_ratio)[:5], ' inl.ratio')
            if dataset[0]=="v":
                list_truematch_number_view.append(plain_indxs_in1.size(0))
            elif dataset[0]=="i":
                list_truematch_number_light.append(plain_indxs_in1.size(0))
    
    return list_truematch_number_view, list_truematch_number_light           

hpatches_sequence_path = r"/home/chen/data/hpatches-sequences-release/"

list_true_matches_v, list_true_matches_light = test_on_hpatchsequence(hpatches_sequence_path)

tml = np.asarray(list_true_matches_light)
tmv = np.asarray(list_true_matches_v)

mean_all = np.mean(np.concatenate((tml,tmv)))
mean_l = np.mean(tml)
mean_v = np.mean(tmv)
print('Average correct match Viewpoint: ', mean_v) 
print('Average correct match Illumination: ', mean_l) 
print('Average correct match for all: ', mean_all) 