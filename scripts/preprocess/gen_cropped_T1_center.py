#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

def calc_crop_indices(input_shape, output_shape):
    x = ((input_shape[0] - output_shape[0]) // 2, (input_shape[0] + output_shape[0]) // 2)
    y = ((input_shape[1] - output_shape[1]) // 2, (input_shape[1] + output_shape[1]) // 2)
    z = ((input_shape[2] - output_shape[2]) // 2, (input_shape[2] + output_shape[2]) // 2)
    return x, y, z

def main():
    subj_list_path = sys.argv[1]
    subj_list = pd.read_csv(subj_list_path, header=None)[0].values

    img_dir = sys.argv[2]
    out_dir = sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    input_shape = (193, 229, 193)
    output_shape = (160, 192, 160)
    x, y, z = calc_crop_indices(input_shape, output_shape)

    for sid in subj_list:
        img_path = os.path.join(img_dir, f"{sid}_ses-01_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz")
        img_name = os.path.basename(img_path)
        img = nib.load(img_path)
        
        if img.shape != input_shape:
            print(f"[WARN] image shape for {sid} is {img.shape}, expected {input_shape}. Skipping.")
            continue

        print(f"[INFO] generating cropped image for {sid}")
        cropped_img = img.slicer[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        nib.save(cropped_img, os.path.join(out_dir, img_name))

if __name__ == "__main__": 
    main()