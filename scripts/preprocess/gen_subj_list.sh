#!/usr/bin/env bash

out_file=$1
if [[ -f $out_file ]]; then rm $out_file; fi

t1_copy_dir=$2
if [[ ! -d $t1_copy_dir ]]; then mkdir -p $t1_copy_dir; fi

t1_img_suffix=$3

for subj_dir in $4/sub-*/; do
	sid=$(basename "${subj_dir}")
	t1_file="${subj_dir}/ses-01/anat/${sid}${t1_img_suffix}"
	
	if [[ -f $t1_file ]]; then
		echo $sid >> $out_file
		
		if [[ ! -f $t1_copy_dir/$(basename "${t1_file}") ]]; then
			echo "[INFO] copying T1w image for ${sid}"
			cp $t1_file $t1_copy_dir
		else
			echo "[SKIP] T1w image has been copied for ${sid}"
		fi
	fi
done