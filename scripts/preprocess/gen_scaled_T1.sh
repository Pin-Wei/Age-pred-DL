#!/usr/bin/env bash
set -euo pipefail

in_dir=$2
out_dir=$3
if [[ ! -d $out_dir ]]; then mkdir -p $out_dir; fi

for sid in `cat $1`; do
	file_name="${sid}_ses-01_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz"
	
    if [[ ! -f $out_dir/$file_name ]]; then
		echo "[INFO] generating scaled T1w image for ${sid}"
		ImageMath 3 $out_dir/$file_name \
			RescaleImage $in_dir/$file_name 0 1
	else
		echo "[SKIP] scaled T1w image has been generated for ${sid}"
    fi
done