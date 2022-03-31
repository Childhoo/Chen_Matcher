
# Variant: BSS
python Thesis_test_affnet_match_features_isprsimageblocks.py --aff_type=Baumberg --orinet_type=SIFT --desc_type=SIFT --method_name=HBSS --GT_match_threshold=1.0 

# Variant: BSH
python Thesis_test_affnet_match_features_isprsimageblocks.py --descriptor_model_weights=pretrained/hardnetBr6.pth --aff_type=Baumberg --orinet_type=SIFT --desc_type=HardNet --method_name=HBSH --GT_match_threshold=1.0

# Variant: BSW
python Thesis_test_affnet_match_features_isprsimageblocks.py --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=Baumberg --orinet_type=SIFT --desc_type=WeMNet --method_name=HBSW 

# Variant:AOH
python Thesis_test_affnet_match_features_isprsimageblocks.py --weightd_fname_affnet=pretrained/AffNet.pth --weightd_fname_orinet=pretrained/OriNet.pth --descriptor_model_weights=pretrained/hardnetBr6.pth --aff_type=AffNet --orinet_type=OriNet --desc_type=HardNet --method_name=HAOH-Brown 

# Variant: AOW
python Thesis_test_affnet_match_features_isprsimageblocks.py --weightd_fname_affnet=pretrained/AffNet.pth --weightd_fname_orinet=pretrained/OriNet.pth --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=AffNet --orinet_type=OriNet --desc_type=WemNet --method_name=HAOW-Brown --GT_match_threshold=1.0

# Variant:MMW
python Thesis_test_affnet_match_features_isprsimageblocks.py --weightd_fname_affnet=./logs/MoNet_lr005_10M_20ep_aswap_05012021_Brown6_withvalidation_AffNetFastHardNet_0.005_12000000_HardNet/checkpoint_9.pth --weightd_fname_orinet=logs/MGNet_lr005_10M_20ep_aswap_04012021_Brown6_withvalidation_OriNet_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_9.pth  --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --aff_type=AffNet --orinet_type=MoNet --desc_type=WemNet --method_name=HMMW-Brown --GT_match_threshold=1.0

# Variant: FuW
python Thesis_test_affnet_match_features_isprsimageblocks.py  --weightd_fname_affnet=./logs/fullaff_6brown_lr005_12M_40ep_aswap_05012021B_moreepochs_HardNetLoss_WeakMatchHardnet_Momentum_ratio_skew_MeanGradDir_AffNetFast4RotNosc_6Brown_HardNet_0.005_12000000_HardNet/checkpoint_39.pth --descriptor_model_weights=logs/DescNetNet_BadSampler_WeakMatch_lr005_30M_20ep_aswap_13082020_5.0WeakLoss_6Brown_AffNetFast_6Brown_HardNet_0.05_30000000_HardNet/checkpoint_desc_9.pth --method_name=HFuW-Brown --aff_type=FullAffine --desc_type=WemNet --aff_type=FullAffine --GT_match_threshold=1.0

# Variant: BSS
python Thesis_test_affnet_match_features_isprsimageblocks.py --aff_type=Baumberg --orinet_type=SIFT --desc_type=SIFT --method_name=HBSS --GT_match_threshold=1.0 run_on_tile=False

python Matcher_run.py --aff_type=Baumberg --orinet_type=SIFT --desc_type=SIFT --method_name=HBSS --GT_match_threshold=1.0 --run_on_tile=False --image_directory=E://eval_image_blocks/isprsdata/block1/images --base_directory=E://eval_image_blocks/isprsdata/block1 --img_suffix_type=.tif 
