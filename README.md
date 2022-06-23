# Related Paper
This repository is the project code for the following paper:

*Chen, Lin, and Christian Heipke. "Deep learning feature representation for image matching under large viewpoint and viewing direction change." ISPRS Journal of Photogrammetry and Remote Sensing 190 (2022): 94-112.*

which is a compact version of the following thesis:

*Chen, L. (2021): Deep Learning for Feature based Image Matching. Wissenschaftliche Arbeiten der Fachrichtung Geodäsie und Geoinformatik der Leibniz Universität Hannover, ISSN 0065-5325, Nr. 369), Dissertation, Hannover, 2021. (paper form)*

*Chen, Lin: Deep learning for feature based image matching. München 2021. Deutsche Geodätische Kommission: C (Dissertationen): Heft Nr. 867. https://publikationen.badw.de/de/047233519 (digital form)*

Please note that this repository is built based on affnet (see affnet here: https://github.com/ducha-aiki/affnet)

Please kindly cite the above journal paper / PhD dissertation if you employ this repository.

# How to install this repo?
Python 3.6 or higher version is required

The required packages for running the code are:
- matplotlib
- tqdm
- opencv-python
- scipy
- seaborn

to install the above packages, run

```pip install -r requirements.txt``` 

pytorch 0.4.1/1.7.1 (other pytorch versions should also be possible)

An examplar install command for pytorch 1.7.1 with cuda 11.0 is:

```pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```

For more information about how to install torch, please refer to pytorch official guide https://pytorch.org/get-started/previous-versions/

# Main script: Matcher_run.py
**Input:** 
- images in the same format (e.g., .jpg, .tif) stored under a same folder. This folder is controlled by the command argument "image_directory"

**Process steps:** 
1. detect Hessian feature
2. estimate feature affine shape and assign feature orientation [differnt variants possible]
3. compute feature descriptors. 
4. nearest neigbour / NN ratio based feature matching

**Output:**
- feature file for each input image [position_x, position_y, scale, orientation, 128D descriptor]
- raw feature matching results. 
The raw matching file is in the format of: left image, right image and then each row represents the matching pair composed of [feature_ind on left image, feature_ind on right image]

# Usage Examples

The important arguments for run the command line are:
- image_directory (dir where images are stored) 
- base_directory (dir when result files [detected features and descriptor, matching files] should be stored)
- img_suffix_type (img format, e.g., .jpg, .tif)

Apart from the above three arguments, the other important arguments are:
- aff_type (which type of affine shape estimation is used)
- orinet_type (which type of orientation assignment is used)
- desc_type (which type of descriptor is used)
- method_name (a name for the method in running)
- weightd_fname_affnet (model weights for learned affine shape estimation network)
- weightd_fname_orinet (model weights for learned orientation network)
- descriptor_model_weights (model weights for learned descriptors)
- run_on_tile (whether to run processing on images tiles, if True, then run on 2x2 tiles and then combine weights)


To run different variants in the paper, please refer to the following example cases:
### Variant: BSS
```python Matcher_run.py --aff_type=Baumberg --orinet_type=SIFT --desc_type=SIFT --method_name=HBSS --GT_match_threshold=1.0 --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix```

### Variant: BSH
```python Matcher_run.py --descriptor_model_weights=pretrained/hardnetBr6.pth --aff_type=Baumberg --orinet_type=SIFT --desc_type=HardNet --method_name=HBSH --GT_match_threshold=1.0 --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix```

### Variant: BSW
```python Matcher_run.py --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=Baumberg --orinet_type=SIFT --desc_type=WeMNet --method_name=HBSW --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix``` 

### Variant:AOH
```python Matcher_run.py --weightd_fname_affnet=pretrained/AffNet.pth --weightd_fname_orinet=pretrained/OriNet.pth --descriptor_model_weights=pretrained/hardnetBr6.pth --aff_type=AffNet --orinet_type=OriNet --desc_type=HardNet --method_name=HAOH-Brown --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix``` 

### Variant: AOW
```python Matcher_run.py --weightd_fname_affnet=pretrained/AffNet.pth --weightd_fname_orinet=pretrained/OriNet.pth --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=AffNet --orinet_type=OriNet --desc_type=WemNet --method_name=HAOW-Brown --GT_match_threshold=1.0 --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix```

### Variant:MMW
```python Matcher_run.py --weightd_fname_affnet=./logs/MoNet_lr005_10M_20ep_aswap_05012021_Brown6_withvalidation_AffNetFastHardNet_0.005_12000000_HardNet/checkpoint_9.pth --weightd_fname_orinet=logs/MGNet_lr005_10M_20ep_aswap_04012021_Brown6_withvalidation_OriNet_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_9.pth  --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=AffNet --orinet_type=MoNet --desc_type=WemNet --method_name=HMMW-Brown --GT_match_threshold=1.0 --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix```

### Variant: FuW
```python Matcher_run.py --weightd_fname_affnet=./logs/fullaff_6brown_lr005_12M_40ep_aswap_05012021B_moreepochs_HardNetLoss_WeakMatchHardnet_Momentum_ratio_skew_MeanGradDir_AffNetFast4RotNosc_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_39.pth --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --method_name=HFuW-Brown --aff_type=FullAffine --desc_type=WemNet --aff_type=FullAffine --GT_match_threshold=1.0 --image_directory=your image path --base_directory=your base path for storing results --img_suffix_type=image suffix```





