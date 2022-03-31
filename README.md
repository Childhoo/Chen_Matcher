# Related Paper
This repository is the project code for the following paper:

Lin Chen, Christian Heipke, Deep Learning Feature Representation for Image Matching under Large Viewpoint and Viewing Direction Change. (Under review)

which is a compact version of the following thesis:

Chen, L. (2021): Deep Learning for Feature based Image Matching. Wissenschaftliche Arbeiten der Fachrichtung Geodäsie und Geoinformatik der Leibniz Universität Hannover, ISSN 0065-5325, Nr. 369), Dissertation, Hannover, 2021. (paper form)

Chen, Lin: Deep learning for feature based image matching. München 2021. Deutsche Geodätische Kommission: C (Dissertationen): Heft Nr. 867. https://publikationen.badw.de/de/047233519 (digital form)

Please note that this repository is built based on affnet (see affnet here: https://github.com/ducha-aiki/affnet)

# Required packages

The required packages for running the code are:
matplotlib
tqdm
opencv-python
scipy
seaborn

to install the above packages, run 
```pip install -r requirements.txt``` 

pytorch 0.4.1/1.7.1 (other pytorch versions should also be possible)

An example install command for pytorch 1.7.1 with cuda 11.0 is:
```pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```

To install torch, please refer to pytorch official guide https://pytorch.org/get-started/previous-versions/

# Usage Examples

Input: images in the same format (e.g., .jpg, .tif) stored under a same folder
Process: detect Hessian feature, estimate feature affine shape and assign feature orientation, compute feature descriptors. Afterwards, nearest neigbour based feature matching is also conducted.
Output: feature file and raw feature matching results.

## run networks to extract features and descriptors, and then match
To run different variants in the paper, please refer to run_affnet_orinet_on_ISPRS_aerialblocks.cmd
#### Variant: BSS
```python Thesis_test_affnet_match_features_isprsimageblocks.py --aff_type=Baumberg --orinet_type=SIFT --desc_type=SIFT --method_name=HBSS --GT_match_threshold=1.0```

#### Variant: BSH
```python Thesis_test_affnet_match_features_isprsimageblocks.py --descriptor_model_weights=pretrained/hardnetBr6.pth --aff_type=Baumberg --orinet_type=SIFT --desc_type=HardNet --method_name=HBSH --GT_match_threshold=1.0```

#### Variant: BSW
python Thesis_test_affnet_match_features_isprsimageblocks.py --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=Baumberg --orinet_type=SIFT --desc_type=WeMNet --method_name=HBSW 

#### Variant:AOH
python Thesis_test_affnet_match_features_isprsimageblocks.py --weightd_fname_affnet=pretrained/AffNet.pth --weightd_fname_orinet=pretrained/OriNet.pth --descriptor_model_weights=pretrained/hardnetBr6.pth --aff_type=AffNet --orinet_type=OriNet --desc_type=HardNet --method_name=HAOH-Brown 

#### Variant: AOW
python Thesis_test_affnet_match_features_isprsimageblocks.py --weightd_fname_affnet=pretrained/AffNet.pth --weightd_fname_orinet=pretrained/OriNet.pth --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=AffNet --orinet_type=OriNet --desc_type=WemNet --method_name=HAOW-Brown --GT_match_threshold=1.0

#### Variant:MMW
python Thesis_test_affnet_match_features_isprsimageblocks.py --weightd_fname_affnet=./logs/MoNet_lr005_10M_20ep_aswap_05012021_Brown6_withvalidation_AffNetFastHardNet_0.005_12000000_HardNet/checkpoint_9.pth --weightd_fname_orinet=logs/MGNet_lr005_10M_20ep_aswap_04012021_Brown6_withvalidation_OriNet_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_9.pth  --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=AffNet --orinet_type=MoNet --desc_type=WemNet --method_name=HMMW-Brown --GT_match_threshold=1.0

#### Variant: FuW
python Thesis_test_affnet_match_features_isprsimageblocks.py  --weightd_fname_affnet=./logs/fullaff_6brown_lr005_12M_40ep_aswap_05012021B_moreepochs_HardNetLoss_WeakMatchHardnet_Momentum_ratio_skew_MeanGradDir_AffNetFast4RotNosc_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_39.pth --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --method_name=HFuW-Brown --aff_type=FullAffine --desc_type=WemNet --aff_type=FullAffine --GT_match_threshold=1.0





