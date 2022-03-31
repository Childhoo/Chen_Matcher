# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 10:44:35 2020

@author: Lin Chen

match the features generated from learned aff modules

including:
    1 aff+ori+desc
    2 fullaff+desc

for the first one, the orientation module can be switched off.
aff/ori can be AffNet/OriNet or MGNet/MoNet

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
AffNetFast52RotUp,AffNetFast52Rot,AffNetFast5Rot, AffNetFast4Rot, AffNetFast4Rot, OriNetFast, OriNetFast_Quaternion
from architectures import AffNetFast2Par,AffNetFastBias
from pytorch_sift import SIFTNet
from HardNet import HardNet, L2Norm
#from Losses import loss_HardNetDetach, loss_HardNet
from Losses import loss_HardNet
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A,visualize_LAFs
import seaborn as sns
from Losses import distance_matrix_vector
from ReprojectionStuff import get_GT_correspondence_indexes, reprojectLAFs
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid, NMS3dAndComposeA
import glob

USE_CUDA = True
parser = argparse.ArgumentParser(description='PyTorch AffNet')
parser.add_argument('--image_directory', type=str,
                    default='',
                    help='path to the directory of images')
parser.add_argument('--base_directory', type=str,
                    default='',
                    help='path to the base directory')
parser.add_argument('--weightd_fname_affnet', type=str,
                    default='./logs/aff_lr005_12M_20ep_aswap_13092020B_moreepochs_HardNetLoss_WeakMatchHardnet_Momentum_ratio_skew_MeanGradDir_AffNetFast4RotNosc_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_39.pth',
                    help='path to affnet weights')
parser.add_argument('--weightd_fname_orinet', type=str,
                    default='pretrained/OriNet.pth',
                    help='path to orientation part weight')
parser.add_argument('--descriptor_model_weights', type=str,
                    default="logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth",
                    help='path to descriptor weights')
parser.add_argument('--method_name', type=str,
                    default="AffNet-HardNet",
                    help='path to descriptor weights')
parser.add_argument('--aff_type', type=str,
                    default="FullAffine",
                    help='type of involved affine modules')
parser.add_argument('--orinet_type', type=str,
                    default="OriNet",
                    help='type of involved orientation module')
parser.add_argument('--desc_type', type=str,
                    default="HardNet",
                    help='type of involved descriptors')
parser.add_argument('--img_suffix_type', type=str,
                    default=".jpg",
                    help='type of image file format, e.g., .tif, .jpg..')
parser.add_argument('--run_on_tile', type=str,
                    default=True,
                    help='whether to run the feature detection and description on image tile. True means run on tile.')
parser.add_argument('--GT_match_threshold', type=float,
                    default=3.0,
                    help='ground truth match threshold in pixel')
args = parser.parse_args()

### Initialization of affnet, orinet and descriptor network

#different choices: SIFT, MoNet, OriNet
if args.orinet_type == "SIFT":
    OriNetPix = OrientationDetector(patch_size = 19)
elif args.orinet_type == "MoNet":
    OriNetPix = OriNetFast_Quaternion(PS=32)
    checkpoint_orinet = torch.load(args.weightd_fname_orinet)
    OriNetPix.load_state_dict(checkpoint_orinet['state_dict'])
    OriNetPix.eval()
else:
    OriNetPix = OriNetFast(PS=32)
    checkpoint_orinet = torch.load(args.weightd_fname_orinet)
    OriNetPix.load_state_dict(checkpoint_orinet['state_dict'])
    OriNetPix.eval()

if args.desc_type == "SIFT":
    descriptor = SIFTNet(patch_size=32)
else:
    descriptor = HardNet()
    hncheckpoint = torch.load(args.descriptor_model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.eval() #whether to use eval mode or not, is a good question?


if args.aff_type == "FullAffine":
    AffNetPix = AffNetFast4RotNosc(PS = 32)
    checkpoint = torch.load(args.weightd_fname_affnet)
    AffNetPix.load_state_dict(checkpoint['state_dict'])
    AffNetPix.eval()
elif args.aff_type == "AffNet":
    AffNetPix = AffNetFast(PS = 32)
    checkpoint = torch.load(args.weightd_fname_affnet)
    AffNetPix.load_state_dict(checkpoint['state_dict'])
    AffNetPix.eval()
elif args.aff_type == "Baumberg":
    AffNetPix = AffineShapeEstimator(patch_size=19)
else:
    print("unknown affnet type, please check!")

#construct the feature detector with affnet and orinet as input
detector = ScaleSpaceAffinePatchExtractor( mrSize = 6.0, num_features = 3000,
                                          border = 5, num_Baum_iters = 1, 
                                          AffNet = AffNetPix, OriNet = OriNetPix)
if USE_CUDA:
    detector = detector.cuda()
    descriptor = descriptor.cuda()

def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    if USE_CUDA:
        var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape

def get_geometry_and_descriptors(img, det, desc, do_ori = True, do_scale_check=True):
    with torch.no_grad():
        LAFs, resp = det(img, do_ori = do_ori, do_scale_check = do_scale_check)
        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = desc(patches)
    return LAFs, descriptors

def get_geometry_and_descriptors_response(img, det, desc):
    with torch.no_grad():
        LAFs, resp = det(img, do_ori = True)
        patches = detector.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = descriptor(patches)
    return LAFs, descriptors, resp


def detect_feature_descriptor_from_tile(img, detector, descriptor,\
                                        wid_num=2, heig_num = 2,):
    width_step = int(img.shape[3]/2)
    height_step = int(img.shape[2]/2)
    for ii in range(wid_num):
        for jj in range(heig_num):
            wid_begin = ii*width_step
            heig_begin = jj*height_step
            cur_img = img[:,:,heig_begin:heig_begin+height_step,wid_begin:wid_begin+width_step]
            if args.aff_type == "FullAffine":
                work_LAFs, descriptors = get_geometry_and_descriptors(cur_img, detector, descriptor, do_ori=False, do_scale_check=False)
            else:
                work_LAFs, descriptors = get_geometry_and_descriptors(cur_img, detector, descriptor, do_ori=True, do_scale_check=True)
            #img_tiles.append(img[:,:,heig_begin:heig_begin+height_step,wid_begin:wid_begin+width_step])
            work_LAFs[:,0,2] += wid_begin
            work_LAFs[:,1,2] += heig_begin
            
            if ii==0 and jj==0:
                LAFs = work_LAFs
                desc = descriptors
#                res_all = res
            else:
                LAFs = torch.cat((LAFs, work_LAFs),0)
                desc = torch.cat((desc, descriptors),0)
#                res_all = torch.cat((res_all, res),0)
            
    return LAFs, desc #, res_all


#directory = r"E:/eval_image_blocks/isprsdata/only_nadir/block3/images/"
#directory_base = r"E:/eval_image_blocks/isprsdata/only_nadir/block3/"
directory = args.image_directory
directory_base = args.base_directory

all_desc=[] 
all_feat=[] #list to store all features/descriptors for later matching stage

if not os.path.exists(os.path.join(directory_base, "features_computed")):
    os.mkdir(os.path.join(directory_base, "features_computed")) 
    
output_feature_dir = os.path.join(directory_base, "features_computed", args.method_name)
if not os.path.exists(output_feature_dir):
    os.mkdir(output_feature_dir) 

for filename in glob.glob(os.path.join(directory, "*" + args.img_suffix_type)):
    img = load_grayscale_var(filename)
    
    if args.run_on_tile:
        work_LAFs, descriptors = detect_feature_descriptor_from_tile(img, detector, descriptor)
    else:
        if args.aff_type == "FullAffine":
            work_LAFs, descriptors = get_geometry_and_descriptors(img, detector, descriptor, do_ori=False, do_scale_check=False)
        else:
            work_LAFs, descriptors = get_geometry_and_descriptors(img, detector, descriptor, do_ori=True, do_scale_check=True)

#    print(work_LAFs.shape)
#    print(descriptors.shape)
    
    
    #convert to normal form
    LAFs = np.zeros((work_LAFs.shape[0], 6))
    work_LAFs = work_LAFs.cpu().numpy()
    LAFs[:,0] = work_LAFs[:,0,2]
    LAFs[:,1] = work_LAFs[:,1,2] 
    LAFs[:,2] = work_LAFs[:,0,0]
    LAFs[:,3] = work_LAFs[:,0,1]
    LAFs[:,4] = work_LAFs[:,1,0]
    LAFs[:,5] = work_LAFs[:,1,1]
    
    all_feat.append(LAFs)
    all_desc.append(descriptors) #store features and descriptors here
    

    out_file_ori_name = output_feature_dir + "/" + os.path.basename(filename) + ".txt"
    
    #convert the descritors to range [0, 255]
    if np.max(np.abs(descriptors.cpu().numpy())) > 0.5:
        print("note!, there is wrong value here")
        #truncate value of descriptor to 0.5
        desc_or = descriptors.cpu().numpy()
        desc_or[np.abs(desc_or)>0.5] = 0.49
        desc_n = np.multiply(desc_or,256) + 128
    else:
        desc_n = np.multiply(descriptors.cpu().numpy(),256) + 128
    desc_normalized = desc_n.astype(int)
    if np.max(descriptors.cpu().numpy()) > 0.5:
        print("note!, there is wrong value here")
    
    
    with open(out_file_ori_name, 'a') as the_file:
        the_file.write('{} {}\n'.format(len(LAFs),desc_normalized.shape[1]))
        for line in range(len(LAFs)):
            np.savetxt(the_file, LAFs[line,0:2], fmt='%.6f ', newline='',)
            the_file.write('2.0')
            the_file.write(' 0.0')
            np.savetxt(the_file, desc_normalized[line,:], fmt=' %d', newline='',) #change this to cope with colmap
            if line!=len(LAFs)-1:
                the_file.write('\n')  




img_names =  [os.path.basename(x) for x in glob.glob(os.path.join(directory, "*" + args.img_suffix_type))]
num_images = len(img_names)
# after extracting feature/descriptors, match the extracted features

# Ratio_Threshold = min_dist, idxs_in_2 = torch.min(dist_matrix,1)
matching_strategy = "ratio"  #or "nn_distance"
Distance_threshold = 0.8
SNN_threshold = 0.85
out_file_match_name = directory_base + args.method_name + matching_strategy + "_matches.txt"


with open(out_file_match_name, 'a') as the_file:
    for i in np.arange(len(all_desc)-1):
        for j in np.arange(i+1,len(all_desc)):
            descriptors1 = all_desc[i]
            descriptors2 = all_desc[j]
            dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
            min_dist, idxs_in_2 = torch.min(dist_matrix,1)                      
            
            if matching_strategy == "ratio":
                dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
                min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
                mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
            elif matching_strategy == "nn_distance": #nearest neigbour distance threshold
                mask = min_dist  <= Distance_threshold
            
            tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False).cuda()[mask]
            tent_matches_in_2 = idxs_in_2[mask]
            tent_matches_in_1 = tent_matches_in_1.cpu().numpy()
            tent_matches_in_2 = tent_matches_in_2.cpu().numpy()                        
            
#            np.savetxt(the_file, img_names[i], img_names[j])
            the_file.write(img_names[i])
            the_file.write(' ')
            the_file.write(img_names[j])
            the_file.write('\n')
            for n in np.arange(tent_matches_in_1.shape[0]):
#                np.savetxt(the_file, tent_matches_in_1[n], tent_matches_in_2[n])
                the_file.write(str(tent_matches_in_1[n]))
                the_file.write(' ')
                the_file.write(str(tent_matches_in_2[n]))
                the_file.write('\n')
            
            #after writing matching of each pair, go further or stop?
            if i==num_images-2 and j==num_images-1:
                pass
#                print('done!')
            else:
                the_file.write('\n')
                
        print("ratio based matching done! for image", i)
            
    
matching_strategy = "nn_distance"  #or "nn_distance"
SNN_threshold = 0.85
out_file_match_name = directory_base + args.method_name + matching_strategy + "_matches.txt"


with open(out_file_match_name, 'a') as the_file:
    for i in np.arange(len(all_desc)-1):
        for j in np.arange(i+1,len(all_desc)):
            descriptors1 = all_desc[i]
            descriptors2 = all_desc[j]
            dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
            min_dist, idxs_in_2 = torch.min(dist_matrix,1)                      
            
            if matching_strategy == "ratio":
                dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
                min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
                mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
            elif matching_strategy == "nn_distance": #nearest neigbour distance threshold
                mask = min_dist  <= Distance_threshold
            
            tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False).cuda()[mask]
            tent_matches_in_2 = idxs_in_2[mask]
            tent_matches_in_1 = tent_matches_in_1.cpu().numpy()
            tent_matches_in_2 = tent_matches_in_2.cpu().numpy()                        
            
#            np.savetxt(the_file, img_names[i], img_names[j])
            the_file.write(img_names[i])
            the_file.write(' ')
            the_file.write(img_names[j])
            the_file.write('\n')
            for n in np.arange(tent_matches_in_1.shape[0]):
#                np.savetxt(the_file, tent_matches_in_1[n], tent_matches_in_2[n])
                the_file.write(str(tent_matches_in_1[n]))
                the_file.write(' ')
                the_file.write(str(tent_matches_in_2[n]))
                the_file.write('\n')
            
            #after writing matching of each pair, go further or stop?
            if i==num_images-2 and j==num_images-1:
                pass
#                print('done!')
            else:
                the_file.write('\n')
                
        print("NN based matching done! for image", i)