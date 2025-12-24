#!/usr/bin/env bash
set -euo pipefail

out_dir=$2
if [[ ! -d $out_dir ]]; then mkdir -p $out_dir; fi

prep_dir=$3

for sid in `cat $1`; do
	anat_file="${prep_dir}/${sid}/ses-01/anat/${sid}_ses-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    mask_file="${prep_dir}/${sid}/ses-01/anat/${sid}_ses-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    out_file="${out_dir}/${sid}_ses-01_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz"

    if [[ ! -f $out_file ]]; then

		if [[ ! -f $anat_file || ! -f $mask_file ]]; then
			echo "[WARN] missing input for ${sid}"
			continue
		fi

		echo "[INFO] generating masked T1w image for ${sid}"
		ImageMath 3 $out_file m $anat_file $mask_file
		
	else
		echo "[SKIP] masked T1w image has been generated for ${sid}"
	fi
done