#!/usr/bin/env bash

current_dir=`cwd`

template=$1
native_img=$2
mid_dir=$3
out_dir=$4
out_img_suffix=$5

if [[ ! -d $out_dir ]]; then mkdir -p $out_dir; fi
if [[ ! -d $mid_dir ]]; then mkdir -p $mid_dir; fi

for sid in `cat $6`; do
	out_img="${sid}${out_img_suffix}"

	if [[ ! -f $out_dir/$out_img ]]; then
		if [[ ! -f $mid_dir/"${sid}_Warped.nii.gz" ]]; then
			echo "[INFO] performing registration for ${sid}"
			cd $mid_dir
			antsRegistrationSyN.sh -d 3 -n 16 \
				-f $template \
				-m "${native_img/XXX/$sid}" \
				-o "${sid}_"
		fi
		cp $mid_dir/"${sid}_Warped.nii.gz" $out_dir/$out_img
	else
		echo "[SKIP] standardized T1w image has been generated for ${sid}"
    fi
done

cd $current_dir
