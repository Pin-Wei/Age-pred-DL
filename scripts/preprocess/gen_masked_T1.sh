#!/usr/bin/env bash
set -euo pipefail

out_dir=$1
if [[ ! -d $out_dir ]]; then mkdir -p $out_dir; fi

prep_dir=$2
anat_suffix=$3
mask_suffix=$4
out_suffix=$5

for sid in `cat $6`; do
	anat_file="${prep_dir}/${sid}/ses-01/anat/${sid}${anat_suffix}"
    mask_file="${prep_dir}/${sid}/ses-01/anat/${sid}${mask_suffix}"
    out_file="${out_dir}/${sid}${out_suffix}"

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