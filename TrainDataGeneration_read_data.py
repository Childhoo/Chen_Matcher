# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:54:16 2019

@author: chenlin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:47:26 2017
@author: lin chen
this files reads the Brown Dataset and generates patch pairs of matched and non-matched 
features for learning descriptors using CNN
"""
import tensorflow as tf
#using tf.train.match_filenames_once() function to match all the images and put them somewhere..
#import os
import glob
from PIL import Image
import random

import numpy as np

import matplotlib.pyplot as plt
import time

path = "C:\\Tensorflow\\notredame"

#files = os.listdir(path)
#acquire all the .bmp file from the path
files = glob.glob(path+'\\*.bmp')


def Read_Info_BrownDataset(path_infofile):
    """Read a Brown Dataset Info file and returns
       
       arg: 
           path_infofile: path for the "info.txt" of dataset
           
       return
          list file for the patch's 3D point index
          list file for the patch's Image index (extracted from which image) 
          list file for the number of patches of each 3D point
    """
    f = open(path_infofile, 'r')                  
    result = list()
    Index_3Dpointlist = list()
    Index_Imagelist = list()
    NumPatch4_3Dpoint  = list() # record the number of patches for each 3D point
    Num_Patches = 0
    Num_3Dpoints = 0
    last_number = -1
    Num_Current3Dpoint = 0

    for line in f.readlines():                        
        line = line.strip()
        numbers1, numbers2  = line.split()  #split the str into 2 parts (split by space) 
                     
        if not len(line) or line.startswith('#'):      
            continue                                    
        result.append(line)
        Index_3Dpointlist.append(int(numbers1))
        Index_Imagelist.append(int(numbers2))
        if not int(numbers1)==int(last_number):
            Num_3Dpoints = Num_3Dpoints + 1 #record the number of 3D points
            NumPatch4_3Dpoint.append(Num_Current3Dpoint)
            Num_Current3Dpoint = 1
        else:
            Num_Current3Dpoint = Num_Current3Dpoint + 1
        Num_Patches = Num_Patches + 1
        last_number = numbers1    
    NumPatch4_3Dpoint.append(Num_Current3Dpoint)
    NumPatch4_3Dpoint.pop(0) #remove the first useless value
    return Index_3Dpointlist, Index_Imagelist, NumPatch4_3Dpoint

def checkPatch_index_inImages(Index_check_patch,num_col,num_row,reshaped_Index_Array):
    """check the index of patches and the size of images"""
    aaa = np.where(reshaped_Index_Array==Index_check_patch)
    return aaa[0][0],aaa[1][0],aaa[2][0] 
    #this part cost too much time, it should be somehow changed to save more time 

# this method does not use the np.where() wchich is a time consuming finding process
def checkPatch_index_inImages_naive(Index_check_patch,Num_patches_perImage,num_col,num_row):
    """check the index of patches and the size of images"""
    aaa = int(Index_check_patch/Num_patches_perImage)
    nn = Index_check_patch%Num_patches_perImage
    bbb = int(nn/num_col)
    ccc = nn%num_col
#    aaa = np.where(reshaped_Index_Array==Index_check_patch)
    return aaa,bbb,ccc 
    #this part cost too much time, it should be somehow changed to save more time 

def Crop_Extract_Patch_Pairs(index_patch_img_left,index_patch_img_right,files,patch_size):
    """
    Crop and extract patches using the index of patch pairs generated before
    
    Args:
        index_patch_img_left @param: index for the left patch
        index_patch_img_left @param: index for the right patch
        files : the file list of image files (containing the image patches)
        
    Return:
        the generated patch pair    
    """
    img = Image.open(files[index_patch_img_left[0]])
    patch_size = 64
    start_corner = index_patch_img_left[2]*patch_size, index_patch_img_left[1]*patch_size
    end_corner =  start_corner[0] + patch_size, start_corner[1] + patch_size
    patch_left = img.crop((start_corner[0],start_corner[1],end_corner[0],end_corner[1]))
    
    img = Image.open(files[index_patch_img_right[0]])
    start_corner = index_patch_img_right[2]*patch_size, index_patch_img_right[1]*patch_size
    end_corner =  start_corner[0] + patch_size, start_corner[1] + patch_size
    patch_right = img.crop((start_corner[0],start_corner[1],end_corner[0],end_corner[1]))
    
    patch_pair = Image.new('L',(2*patch_size,patch_size))
    patch_pair.paste(im=patch_left,box=(0,0))
    patch_pair.paste(im=patch_right,box=(patch_size,0))
    return patch_pair



# read and get the list file from info files of patch dataset
path_infofile = "C:\\Tensorflow\\notredame\\info.txt"
Index_3Dpointlist, Index_Imagelist, NumPatch4_3Dpoint = Read_Info_BrownDataset(path_infofile)

# generate positive training data
Postive_Pair= list()
Start_3DPoint_index = list()
Current_sum_Patch = 0
for i in range(len(NumPatch4_3Dpoint)):
    N = NumPatch4_3Dpoint[i]
    for j in range(N):
        for k in range(j+1,N):
            L_i = Current_sum_Patch + j
            R_i = Current_sum_Patch + k
            Postive_Pair.append([L_i, R_i])
            pass 
    Start_3DPoint_index.append(Current_sum_Patch)    
    Current_sum_Patch = Current_sum_Patch + N

# generate the negative training data
Negtive_Pair= list()
Negtive_Pair_3Dpoints= list()
Num_3DPoints = len(NumPatch4_3Dpoint)
for i in range(len(Postive_Pair)):
    L_3DP,R_3DP = random.sample(range(Num_3DPoints),2) # index for left and right 3D points
    L_i = Start_3DPoint_index[L_3DP] + random.sample(range(NumPatch4_3Dpoint[L_3DP]),1)[0]
    # pick a random patch for the picked 3D point index
    R_i = Start_3DPoint_index[R_3DP] + random.sample(range(NumPatch4_3Dpoint[R_3DP]),1)[0]
    Negtive_Pair.append([L_i, R_i])
    Negtive_Pair_3Dpoints.append([L_3DP,R_3DP])
    pass

# show the histogram of 3D points used in negative training pairs
Negative_pairarray = np.asarray(Negtive_Pair_3Dpoints)
plt.hist(Negative_pairarray)
plt.title("Histogram of used 3Dpoint index in negative pairs")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# generate and store the training dataset in files...
Positive_pairarray = np.asarray(Postive_Pair)
Negtive_Pairarray  = np.asarray(Negtive_Pair)
f = open('TrainingData\\PatchPairs\\file_list.txt', 'w')

time_start = time.time()
for patch_index in range(len(Postive_Pair)):
#for patch_index in range(1000):    
    index_patch_img_left = checkPatch_index_inImages_naive(Positive_pairarray[patch_index,0],256,16,16)
    index_patch_img_right = checkPatch_index_inImages_naive(Positive_pairarray[patch_index,1],256,16,16)
    # check the index from image files and take them out from dataset..   
    patch_image = Crop_Extract_Patch_Pairs(index_patch_img_left,index_patch_img_right,files,64)
    PosPath = "TrainingData\\PatchPairs\\PosPairs\\PosPair" + str(patch_index) + ".jpg"
    patch_image.save(PosPath)
    
    index_patch_img_left = checkPatch_index_inImages_naive(Negtive_Pairarray[patch_index,0],256,16,16)
    index_patch_img_right = checkPatch_index_inImages_naive(Negtive_Pairarray[patch_index,1],256,16,16)
    # check the index from image files and take them out from dataset..   
    patch_image = Crop_Extract_Patch_Pairs(index_patch_img_left,index_patch_img_right,files,64)
    NegPath = "TrainingData\\PatchPairs\\NegPairs\\NegPair" + str(patch_index) + ".jpg"
    patch_image.save(NegPath)
    
    f.write(PosPath + " 1\n")
    f.write(NegPath + " 0\n")
    
    if patch_index%1000 == 0:
        print("proceed:" , 2*patch_index, "/",2*len(Postive_Pair))
    pass
# 
f.close()
time_end = time.time()
print("used time for generating and writing patch pairs in seconds: ",time_end-time_start)












#random.shuffle()