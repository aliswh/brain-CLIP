"""
Image and text processing functions.
"""

import nibabel as nib
import numpy as np

def crop_image(img_path):
    """Crop image to pretrained r3d_18 input format."""
    img = nib.load(img_path)
    img_data = img.get_fdata()
    img_shape = img_data.shape

    # Calculate the center of the image
    center = np.array(img_shape) / 2
    # Calculate the starting and ending point for the crop
    start = np.round(center - np.array([56, 56, 8])).astype(int)
    end = start + np.array([112, 112, 16])

    # Make sure the crop is within the bounds of the image
    start = np.maximum(start, 0)
    end = np.minimum(end, img_shape)

    # Crop the image
    cropped_img = img_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    cropped_img = nib.Nifti1Image(cropped_img, img.affine, header=img.header)
    nib.save(cropped_img, img_path)


def preprocess_image(img_path):
    crop_image(img_path)