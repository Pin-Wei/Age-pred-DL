#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

def compute_bbox(data, threshold=0, pad=1):
    coords = np.array(np.where(data > threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    if pad > 0:
        start = np.maximum(start - pad, 0)
        end = np.minimum(end + pad, data.shape)
    return np.array([start, end]).T

def main():
    subj_list_path = sys.argv[1]
    subj_list = pd.read_csv(subj_list_path, header=None)[0].values

    mask_path = sys.argv[2]
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    [x, y, z] = compute_bbox(mask_data)

    img_dir = sys.argv[3]
    out_dir = sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)

    img_suffix = sys.argv[5]

    for sid in subj_list:
        img_path = os.path.join(img_dir, f"{sid}{img_suffix}")
        img_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, img_name)

        if not os.path.exists(out_path):
            if not os.path.exists(img_path):
                print(f"[WARN] image {img_path} does not exist. Skipping.")
                continue
            
            img = nib.load(img_path)

            if img.affine.shape != mask_img.affine.shape or not np.allclose(img.affine, mask_img.affine):
                raise ValueError(f"Affine mismatch for {sid}")

            print(f"[INFO] generating cropped image for {sid}")
            cropped_img = img.slicer[x[0]:x[1], y[0]:y[1], z[0]:z[1]]
            nib.save(cropped_img, os.path.join(out_dir, img_name))

        else:
            print(f"[SKIP] cropped image for {sid} already exists.")

if __name__ == "__main__": 
    main()