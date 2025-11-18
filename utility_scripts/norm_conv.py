#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" TODO: needs cleanup
"""

import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes


def gaussian_blur(data, mask, blur=0.5):
    # normalized convolution of image with mask
    filtr = gaussian_filter(data * mask, sigma=blur)
    weights = gaussian_filter(mask, sigma=blur)
    filtr /= (weights + 1e-8)
    filtr *= mask

    return filtr


sigma_value = 1.0

img = nib.load(sys.argv[1])
ori_data = img.get_fdata().astype(int)

# Create mask of non-zero voxels
mask = np.zeros(ori_data.shape, dtype=np.float32)
mask[ori_data > 0] = 1

# Fill holes for each label
data = np.zeros(ori_data.shape, dtype=float)
for i in np.unique(ori_data)[1:]:
    tmp = np.zeros(ori_data.shape, dtype=np.uint16)
    tmp[ori_data == i] = 1
    tmp = binary_fill_holes(tmp)
    data[tmp > 0] = i

# Normalize data to [0, 1]
data /= data.max()

# Apply gaussian blur with mask
results = gaussian_blur(data.astype(np.float32), mask.astype(np.float32))

# Save results
results = results.astype(np.float32)
nib.save(nib.Nifti1Image(results, img.affine), sys.argv[2])
