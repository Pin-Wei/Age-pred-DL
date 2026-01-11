#!/usr/bin/env bash

union_mask=$1
src_top=$2
img_suffix=$3

if [[ ! -f $union_mask ]]; then
	ImageMath 3 \
		$union_mask max \
		$src_top/sub-*/ses-01/anat/sub-*$img_suffix
	echo "[INFO] ${union_mask} is generated"
else
	echo "[SKIP] ${union_mask} has been generated" 
fi