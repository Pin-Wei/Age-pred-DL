#!/usr/bin/env bash
set -euo pipefail

in_dir=$1
out_dir=$2
if [[ ! -d $out_dir ]]; then mkdir -p $out_dir; fi

img_suffix=$3

for sid in `cat $4`; do
	file_name="${sid}${img_suffix}"
	
    if [[ ! -f $out_dir/$file_name ]]; then
		echo "[INFO] generating scaled T1w image for ${sid}"
		ImageMath 3 $out_dir/$file_name \
			RescaleImage $in_dir/$file_name 0 1
	else
		echo "[SKIP] scaled T1w image has been generated for ${sid}"
    fi
done