# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:03:40 2019

@author: chenlin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:56:06 2019

@author: chenlin
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

USE_CUDA = True

### Initialization
AffNetPix = AffNetFast(PS = 32)
#weightd_fname = '../../pretrained/AffNet.pth'
weightd_fname = '../../logs/AffNetFast_lr005_10M_20ep_aswap_ipidata_AffNetFast_6Brown_HardNet_0.005_10000000_HardNegC/checkpoint_18.pth'

checkpoint = torch.load(weightd_fname)
AffNetPix.load_state_dict(checkpoint['state_dict'])

AffNetPix.eval()

#load orientation net for more training process
from architectures import OriNetFast
OriNetPix = OriNetFast(PS=32)
weightd_fname_orinet = '../../logs/OriNetFast_lr005_10M_20ep_aswap_ipidata_OriNet_6Brown_HardNet_0.005_10000000_HardNet/checkpoint_14.pth'

checkpoint_orinet = torch.load(weightd_fname_orinet)
OriNetPix.load_state_dict(checkpoint_orinet['state_dict'])
OriNetPix.eval()


detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
                                          border = 5, num_Baum_iters = 1, 
                                          OriNet = OriNetPix, AffNet = AffNetPix)
#detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
#                                          border = 5, num_Baum_iters = 1, 
#                                          AffNet = AffNetPix)
descriptor = HardNet()
model_weights = '../../HardNet++.pth'
hncheckpoint = torch.load(model_weights)
descriptor.load_state_dict(hncheckpoint['state_dict'])
descriptor.eval()


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

#Image loading
try:
    input_img_fname1 = 'img/0000.jpg'#sys.argv[1]
    input_img_fname2 = 'img/0009.jpg'#sys.argv[2]
    output_img_fname = 'kpi_match.png'#sys.argv[3]
except:
    print("Cannot find images")
    sys.exit(1)


directory = r"img/graf/"
import glob
for filename in glob.glob(directory+"*.ppm"):
    img = load_grayscale_var(filename)
    work_LAFs, descriptors, res = get_geometry_and_descriptors_response(img, detector, descriptor)
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
    out_file_ori_name = filename+"ori_affnet_ipi.txt"
    np.savetxt(out_file_ori_name, LAFs, fmt='%10.9f')
    
    out_file_desc_name = filename+"desc_affnet_ipi.txt"
    np.savetxt(out_file_desc_name, descriptors.cpu().numpy(), fmt='%10.9f')
    
    out_file_response_name = filename+"response_affnet_ipi.txt"
    np.savetxt(out_file_response_name, res.cpu().numpy(), fmt='%10.9f')
