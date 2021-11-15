# Related Paper
This repository is the project code for the following paper:

Lin Chen, Christian Heipke, Deep Learning Feature Representation for Image Matching under Large Viewpoint and Viewing Direction Change. (Under review)

and it is built on top of the following thesis:

Chen, L. (2021): Deep Learning for Feature based Image Matching. Wissenschaftliche Arbeiten der Fachrichtung Geodäsie und Geoinformatik der Leibniz Universität Hannover, ISSN 0065-5325, Nr. 369), Dissertation, Hannover, 2021. (paper form)

Chen, Lin: Deep learning for feature based image matching. München 2021. Deutsche Geodätische Kommission: C (Dissertationen): Heft Nr. 867. https://publikationen.badw.de/de/047233519 (digital form)

Please note that this repository is built based on affnet (see affnet here: https://github.com/ducha-aiki/affnet)

# Required packages

The required packages for running the code are:
pytorch 0.4.1

For install guide, please refer to affnet repository for further details.

# Usage Examples

Input: images in the same format (e.g., .jpg, .tif) stored under a same folder
Process: detect Hessian feature, estimate feature affine shape and assign feature orientation, compute feature descriptors. Afterwards, nearest neigbour based feature matching is also conducted.
Output: feature file and raw feature matching results.

## run networks to extract features and descriptors, and then match

### Run the (MMG) MoNet-MGNET-WeMNet
Thesis_test_affnet_match_features_isprsimageblocks.py 

### Run the (FuW) FullAffNet-WeMNet
Thesis_test_affnet_match_features_isprsimageblocks.py 