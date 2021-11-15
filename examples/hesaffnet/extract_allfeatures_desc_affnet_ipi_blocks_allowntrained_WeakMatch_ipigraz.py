# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:17:05 2020

@author: chenlin
extract all features 
"""


#This is only siple example. It DOES NOT DO state-of-art image matching, not even close, because it is lacking RANSAC
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A
from Utils import line_prepender
from architectures import AffNetFast
from HardNet import HardNet
from Losses import distance_matrix_vector
USE_CUDA = True

### Initialization
AffNetPix = AffNetFast(PS = 32)
#weightd_fname = '../../pretrained/AffNet.pth'
#weightd_fname = '../../logs/AffNetFast_lr005_10M_20ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNegC/checkpoint_18.pth'
#weightd_fname = '../../logs/aff_lr005_12M_20ep_aswap_07062020_HardNetLoss_WeakMatchHardnet_AffNetFast_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_14.pth'
weightd_fname = '../../logs/aff_lr005_12M_20ep_aswap_27062020_HardNetLoss_WeakMatchHardnet_MainGradDirec_2ndMomentumratio_AffNetFast_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_14.pth'
#weightd_fname = '../../logs/AffNetFast_hardest_k_lr005_10M_25_0715ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNet_k_hardest/checkpoint_24.pth'
checkpoint = torch.load(weightd_fname)
AffNetPix.load_state_dict(checkpoint['state_dict'])

AffNetPix.eval()

#load orientation net for more training process
from architectures import OriNetFast
OriNetPix = OriNetFast(PS=32)
#weightd_fname_orinet = '../../pretrained/OriNet.pth'
#weightd_fname_orinet = '../../logs/OriNetFast_lr005_10M_20ep_aswap_6browndata_02012020_OriNet_6Brown_HardNet_0.005_10000000_HardNet/checkpoint_15.pth'
weightd_fname_orinet = '../../logs/OriNet_lr005_12M_20ep_aswap_26062020_HardNetLoss_onlyMeanGrad_ipiGraz_Corrected_PosDist_OriNet_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_' + str(9) + '.pth'
checkpoint_orinet = torch.load(weightd_fname_orinet)
OriNetPix.load_state_dict(checkpoint_orinet['state_dict'])
OriNetPix.eval()


from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid, NMS3dAndComposeA

aff_handc = AffineShapeEstimator(patch_size=19)
ori_handc = OrientationDetector(patch_size = 19)

detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 5000,
                                  border = 5, num_Baum_iters = 1,
                                  AffNet = AffNetPix,
                                  OriNet = OriNetPix)

# for all own trained module, the orientation network must be correctly set!
#detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
#                                          border = 5, num_Baum_iters = 1, 
#                                          OriNet = OriNetPix, AffNet = AffNetPix)


descriptor = HardNet()
#model_weights = '../../HardNet++.pth'
model_weights = "../../logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_26052020_HardNetLoss_AffNetFast_6Brown_HardNet_0.005_30000000_HardNet/checkpoint_desc_9.pth"
#model_weights = '../../logs/checkpoint_9.pth'
hncheckpoint = torch.load(model_weights)
descriptor.load_state_dict(hncheckpoint['state_dict'])
descriptor.eval()


#from pytorch_sift import SIFTNet
#descriptor = SIFTNet(patch_size=32)
    

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



## Detection and description
def get_geometry_and_descriptors_response(img, det, desc):
    with torch.no_grad():
        LAFs, resp = det(img, do_ori = True)
        patches = detector.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = descriptor(patches)
    return LAFs, descriptors, resp


def detect_feature_descriptor_from_tile(img, detector, descriptor,\
                                        wid_num=2, heig_num = 2,):
    #first divide images into tiles
#    img_tiles = []
    width_step = int(img.shape[3]/2)
    height_step = int(img.shape[2]/2)
    for ii in range(wid_num):
        for jj in range(heig_num):
            wid_begin = ii*width_step
            heig_begin = jj*height_step
            cur_img = img[:,:,heig_begin:heig_begin+height_step,wid_begin:wid_begin+width_step]
            work_LAFs, descriptors, res = get_geometry_and_descriptors_response(cur_img, detector, descriptor)
            #img_tiles.append(img[:,:,heig_begin:heig_begin+height_step,wid_begin:wid_begin+width_step])
            work_LAFs[:,0,2] += wid_begin
            work_LAFs[:,1,2] += heig_begin
            
            if ii==0 and jj==0:
                LAFs = work_LAFs
                desc = descriptors
                res_all = res
            else:
                LAFs = torch.cat((LAFs, work_LAFs),0)
                desc = torch.cat((desc, descriptors),0)
                res_all = torch.cat((res_all, res),0)
            
    return LAFs, desc, res_all

#Image loading
try:
    input_img_fname1 = 'img/0000.jpg'#sys.argv[1]
    input_img_fname2 = 'img/0009.jpg'#sys.argv[2]
    output_img_fname = 'kpi_match.png'#sys.argv[3]
except:
    print("Cannot find images")
    sys.exit(1)

#
directory = r"E:/eval_image_blocks/block3/images/"
#directory = r"E:/eval_image_blocks/isprsdata/block3/images/"
import glob
for filename in glob.glob(directory+"*.tif"):
    img = load_grayscale_var(filename)
    work_LAFs, descriptors, res = get_geometry_and_descriptors_response(img, detector, descriptor)
#    work_LAFs, descriptors, res = detect_feature_descriptor_from_tile(img, detector, descriptor)
    print(work_LAFs.shape)
    print(descriptors.shape)
    
    #convert to normal form
    LAFs = np.zeros((work_LAFs.shape[0], 6))
    LAFs[:,0] = work_LAFs[:,0,2]
    LAFs[:,1] = work_LAFs[:,1,2] 
    LAFs[:,2] = work_LAFs[:,0,0]
    LAFs[:,3] = work_LAFs[:,0,1]
    LAFs[:,4] = work_LAFs[:,1,0]
    LAFs[:,5] = work_LAFs[:,1,1]
    
    # write into .ori and .desc files
    out_file_ori_name = filename+".txt"
#    np.savetxt(out_file_ori_name, LAFs, fmt='%10.9f')
    
#    out_file_desc_name = filename+"desc_affnet_ipi.txt"
#    np.savetxt(out_file_desc_name, descriptors.cpu().numpy(), fmt='%10.9f')
    
    #convert the descritors to 0 and 255
    desc_n = np.multiply(descriptors.cpu().numpy(),256) + 128
    desc_normalized = desc_n.astype(int)
#    np.savetxt(out_file_ori_name, LAFs, fmt='%10.9f')
#    np.savetxt(out_file_ori_name, desc_normalized, fmt='%d')
    
    
    with open(out_file_ori_name, 'a') as the_file:
        the_file.write('{} {}\n'.format(len(LAFs),desc_normalized.shape[1]))
        for line in range(len(LAFs)):
#            np.savetxt(the_file, LAFs[line,0], fmt='%10.6f', newline='',)
#            the_file.write('{}'.format(LAFs[line,0]))
#            the_file.write(' {}'.format(LAFs[line,1]))
            np.savetxt(the_file, LAFs[line,0:2], fmt='%.6f ', newline='',)
            the_file.write('2.0')
            the_file.write(' 0.0')
#            np.savetxt(the_file, LAFs[line,1:4], fmt=' %.6f', newline='',)
#            np.savetxt(the_file, desc_normalized[line,:], fmt=' %.6f', newline='',)
            np.savetxt(the_file, desc_normalized[line,:], fmt=' %d', newline='',) #change this to cope with colmap
            if line!=len(LAFs)-1:
                the_file.write('\n')  
    
#    out_file_response_name = filename+"response_affnet_ipi.txt"
#    np.savetxt(out_file_response_name, res.cpu().numpy(), fmt='%10.9f')

