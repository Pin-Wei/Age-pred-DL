#!/usr/bin/env bash

top_dir="/home/aclexp/pinwei/Age_pred_DL"
orig_dir="${top_dir}/data/fmriprep"
subj_list="${top_dir}/data/meta/subj_list.txt"

out_top="${orig_dir}/../processed"
t1_copied_dir="${out_top}/preproc_MNI_wholebrain_raw"
t1_masked_dir="${t1_copied_dir/raw/masked}"
t1_scaled_dir="${t1_copied_dir/raw/scaled}"
t1_cropped_dir="${t1_copied_dir/raw/cropped}"

bash preprocess/gen_subj_list.sh $subj_list $t1_copied_dir $orig_dir
bash preprocess/gen_masked_T1.sh $subj_list $t1_masked_dir $orig_dir
bash preprocess/gen_scaled_T1.sh $subj_list $t1_masked_dir $t1_scaled_dir
python3 preprocess/gen_cropped_T1_center.py $subj_list $t1_scaled_dir $t1_cropped_dir

## Create a union mask and crop T1w images using the minimum boundary:
# union_mask="${out_top}/union_mask.nii.gz"
# if [[ ! -f $union_mask ]]; then
	# ImageMath 3 \
		# $union_mask max \
		# $orig_dir/sub-*/ses-01/anat/sub-*_ses-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
	# echo "[INFO] ${union_mask} is generated"
# else
	# echo "[SKIP] ${union_mask} has been generated" 
# fi

# t1_cropped_dir_2="${t1_copied_dir/raw/min-cropped}"
# python3 preprocess/gen_cropped_T1_minimum.py $subj_list $union_mask $t1_scaled_dir $t1_cropped_dir_2

