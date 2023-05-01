"""
Image and text processing functions.
"""

import nibabel as nib
import numpy as np
from transformers import DistilBertTokenizer
import torch.nn.functional as F
import torch
import SimpleITK as sitk


def crop_image(img):
    """Crop image to pretrained r3d_18 input format."""
    img_data = sitk.GetArrayFromImage(img)
    img_shape = img_data.shape

    new_spacing = [sz*spc/nsz for sz, spc, nsz in zip(img.GetSize(), img.GetSpacing(), (112,112,16))]



    center = np.array(img_shape) / 2
    # Calculate the starting and ending point for the crop
    start = np.round(center - np.array([8, 56, 56])).astype(int)
    end = start + np.array([16, 112, 112])

    # Make sure the crop is within the bounds of the image
    start = np.maximum(start, 0)
    end = np.minimum(end, img_shape)

    # Crop the image
    cropped_img = img_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    cropped_img = sitk.GetImageFromArray(cropped_img)    
    cropped_img.SetSpacing(img.GetSpacing())
    cropped_img.SetDirection(img.GetDirection())
    cropped_img.SetOrigin(img.GetOrigin())

    return cropped_img

def preprocess_image(img):
    #post_img = crop_image(img)
    post_img = resize_crop(img)
    return post_img


def tokenize(text_batch):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_batch = tokenizer(text_batch, padding="max_length", return_tensors='pt')
    return encoded_batch

def one_hot_encoding(labels):
    al = len(labels)
    encoding = F.one_hot(torch.arange(0, 5)% 3, num_classes=al).float()
    return {l:e for l,e in zip(labels,encoding)}


def register_images(T2, FLAIR, DWI): 
    T2 = sitk.Cast(T2, sitk.sitkFloat32)
    FLAIR = sitk.Cast(FLAIR, sitk.sitkFloat32)
    DWI = sitk.Cast(DWI, sitk.sitkFloat32)

    FLAIR = sitk.Resample(FLAIR, T2)
    DWI = sitk.Resample(DWI, T2)

    transform_type = sitk.VersorRigid3DTransform() # only translation and rotation
    method = sitk.ImageRegistrationMethod()

    reg_param = {
        method.SetMetricAsMattesMutualInformation : {"numberOfHistogramBins":50},
        method.SetMetricSamplingStrategy: 			{"strategy":method.RANDOM},
        method.SetMetricSamplingPercentage: 		{"percentage":0.01},
        method.SetInterpolator: 					{"Interpolator":sitk.sitkNearestNeighbor},
        method.SetOptimizerAsGradientDescent: 		{"learningRate":1.0,
                                                    "numberOfIterations":100,
                                                    "convergenceMinimumValue":1e-6,
                                                    "convergenceWindowSize":10
                                                    },
        method.SetOptimizerScalesFromPhysicalShift: {},
        method.SetInitialTransform: 				{"transform":transform_type}
    }

    for f, kwargs in reg_param.items():
        f(**kwargs) 

    # Get transform matrix and apply it to flipped image
    transform = method.Execute(T2, FLAIR)
    aligned_FLAIR = sitk.Resample(
        FLAIR, transform,
        sitk.sitkNearestNeighbor, 0.0, FLAIR.GetPixelID()
        )
        
    transform = method.Execute(T2, DWI)
    aligned_DWI = sitk.Resample(
        DWI, transform,
        sitk.sitkNearestNeighbor, 0.0, DWI.GetPixelID()
        )

    result_stack = [T2, aligned_FLAIR, aligned_DWI]
    result_stack = [preprocess_image(img) for img in result_stack]

    return result_stack


def resize_crop(image):
    new_size = (112, 112, 16)
    original_size = image.GetSize()
    spacing = image.GetSpacing()
    new_spacing = [old_sz*old_spc/new_sz for old_sz, old_spc, new_sz in zip(original_size, spacing, new_size)]
    resampled = sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkLinear, image.GetOrigin(), new_spacing, image.GetDirection(), 0.0, image.GetPixelID())
    return resampled



