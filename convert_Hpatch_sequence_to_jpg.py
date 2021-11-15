# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:51:28 2019

@author: chenlin
"""

import cv2
import os

hpatches_sequence_path = r"F:/TrainingData/hpatches-sequences-release/hpatches-sequences-release/"
output_path = 'F:/TrainingData/hpatches_seq_jpg/'
folders = os.listdir(hpatches_sequence_path)

for dataset in folders[60:70]:
    input_img_fname1 = os.path.join(hpatches_sequence_path, dataset,'1.ppm')
    input_img_fname3 = os.path.join(hpatches_sequence_path, dataset,'3.ppm')
    input_img_fname6 = os.path.join(hpatches_sequence_path, dataset,'6.ppm')
    
    img1 = cv2.imread(input_img_fname1)
    output_img1_name = os.path.join(output_path, dataset+'1.jpg')
    
    img3 = cv2.imread(input_img_fname3)
    output_img3_name = os.path.join(output_path, dataset+'3.jpg')
    
    img6 = cv2.imread(input_img_fname6)
    output_img6_name = os.path.join(output_path, dataset+'6.jpg')
    
    cv2.imwrite(output_img1_name, img1)
    cv2.imwrite(output_img3_name, img3)
    cv2.imwrite(output_img6_name, img6)