#!/usr/bin/env bash

top_dir="/home/aclexp/pinwei/Age_pred_DL"
src_top="${top_dir}/data/fmriprep"
out_top="${top_dir}/data/processed"

script_dir="${top_dir}/scripts/preprocess"
subj_list="${top_dir}/data/meta/subj_list.txt"

for space in "MNI" "Native" "CN"; do
	t1_copied_dir="${out_top}/preproc_${space}_wholebrain_raw"
	t1_masked_dir="${t1_copied_dir/raw/masked}"
	t1_scaled_dir="${t1_copied_dir/raw/scaled}"
	t1_cropped_dir="${t1_copied_dir/raw/cropped}"
	t1_cropped_dir_2="${t1_copied_dir/raw/min-cropped}"
	
	union_mask="${out_top}/union_${space}_mask.nii.gz"

	case $space in 
		MNI )
			t1_img_suffix="_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
			mask_img_suffix="${t1_img_suffix/preproc_T1w/brain_mask}"
			output_suffix="${mask_img_suffix/mask/T1w}"
			GEN_SUBJ_LIST=true
			GEN_MASKED_T1=true
			GEN_STANDARDIZED_T1=false
			GEN_SCALED_T1=true
			GEN_CROPPED_T1=true
			GEN_UNION_MASK=true
			GEN_MIN_CROPPED_T1=true
			;;
		Native )
			t1_img_suffix="_ses-01_desc-preproc_T1w.nii.gz"
			mask_img_suffix="${t1_img_suffix/preproc_T1w/brain_mask}"
			output_suffix="${mask_img_suffix/mask/T1w}"
			GEN_SUBJ_LIST=true
			GEN_MASKED_T1=true
			GEN_STANDARDIZED_T1=false
			GEN_SCALED_T1=false
			GEN_CROPPED_T1=false
			GEN_UNION_MASK=false
			GEN_MIN_CROPPED_T1=false
			;;
		CN )
			output_suffix="_ses-01_space-CN200_desc-brain_T1w.nii.gz"
			template="${top_dir}/data/CN200/CN200_brain_1mm.nii"
			GEN_SUBJ_LIST=false
			GEN_MASKED_T1=false
			GEN_STANDARDIZED_T1=true
			GEN_SCALED_T1=true
			GEN_CROPPED_T1=true
			GEN_UNION_MASK=false
			GEN_MIN_CROPPED_T1=false
			;;
	esac

	## Copy T1w images to $t1_copied_dir and cat sid to $subj_list:
	if $GEN_SUBJ_LIST; then
		bash $script_dir/gen_subj_list.sh \
			$subj_list \
			$t1_copied_dir \
			$t1_img_suffix \
			$src_top 
	fi
	
	## Apply masks to remove the skull:
	if $GEN_MASKED_T1; then
		bash $script_dir/gen_masked_T1.sh \
			$t1_masked_dir \
			$src_top \
			$t1_img_suffix \
			$mask_img_suffix \
			$output_suffix \
			$subj_list
	fi
	
	if $GEN_STANDARDIZED_T1; then
		bash $script_dir/gen_standardized_T1.sh \
			$template \
			"${out_top}/preproc_Native_wholebrain_masked/XXX_ses-01_desc-brain_T1w.nii.gz" \
			"${out_top}/transform_to_${space}" \
			"${out_top}/preproc_${space}_wholebrain_masked" \
			$output_suffix \
			$subj_list 
	fi

	## Scale images to range 0-1:
	if $GEN_SCALED_T1; then
		bash $script_dir/gen_scaled_T1.sh \
			$t1_masked_dir \
			$t1_scaled_dir \
			$output_suffix \
			$subj_list 
	fi
	
	## Crop images to a specificed shape:
	if $GEN_CROPPED_T1; then
		python3 $script_dir/gen_cropped_T1_center.py \
			$subj_list \
			$t1_scaled_dir \
			$t1_cropped_dir \
			$space \
			$output_suffix
	fi
	
	## Create a union mask:
	if $GEN_UNION_MASK; then
		bash $script_dir/gen_union_mask.sh \
			$union_mask \
			$src_top \
			$mask_img_suffix
	fi
	
	## Crop T1w images using the minimum boundary (of mask):
	if $GEN_MIN_CROPPED_T1; then
		python3 preprocess/gen_cropped_T1_minimum.py \
			$subj_list \
			$union_mask \
			$t1_scaled_dir \
			$t1_cropped_dir_2 \
			$output_suffix
	fi
done