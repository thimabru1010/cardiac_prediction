import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
from numpy import linalg as LA
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti

if __name__ == '__main__':
    #TODO: Aplicar a segmentação em cada um dos niftis
    input_path = 'data/EXAMES/Exames_Separados/ALL/61113_0_5_partes_moles__10.nii.gz'
    # input_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/61113/Nifti/61113_0/2_cardiac_30.nii.gz'
    output_path = 'data/EXAMES/Exames_Separados/ALL_TotalSeg'
    
    # input_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/122932/Nifti/122932_0/3_escore_de_calcio.nii.gz'
    # output_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/122932/Nifti/122932_0'
    
    input_img = nib.load(input_path)#.get_fdata()
    # os.system(f'TotalSegmentator -i {input_path} -o {os.path.join(output_path, "segs3")} -t coronary_arteries')

    # output_img = totalsegmentator(input_img, task='total')
    output_img = nib.load(f'{output_path}/cardio_segs.nii.gz')
    
    print('-'*50)
    output_data = output_img.get_fdata()
    print(output_data.shape)
    output_data[output_data != 0] = 1
    # Find all unique classes (labels) in the segmentation
    # unique_labels = np.unique(output_data)
    # print("Unique labels in the segmentation:", unique_labels)
    
    # dilation of the mask
    kernel = np.ones((10,10), np.uint8)
    cardio_data = cv2.dilate(output_data, kernel, iterations=3)
    
    # Create a new NIfTI image from the modified data
    new_nifti = nib.Nifti1Image(cardio_data, output_img.affine)

    # Save the new NIfTI image
    nib.save(new_nifti, f'{output_path}/cardio_segs_dilated.nii.gz')
    
    count_area = np.zeros((1, 1, cardio_data.shape[2]))
    count_area = LA.norm(cardio_data, axis=(0, 1))
    print(count_area)
    print(count_area.shape)
    max_index = np.argmax(count_area)
    max_slice = cardio_data[:, :, max_index]
    # Calculate the centroid
    # Get the coordinates of all pixels that are 'on'
    coordinates = np.argwhere(max_slice)

    # Calculate the centroid from the mean of the coordinates
    centroid = coordinates.mean(axis=0)
    print(centroid)
    
    circle_mask = np.zeros(max_slice.shape)  
    cv2.circle(circle_mask, (int(centroid[1]), int(centroid[0])), radius=150, color=1, thickness=-1)
    # Get the rectangle circunscribing the circle
    # Get the coordinates of the non-zero pixels in the circle mask
    non_zero_coords = np.argwhere(circle_mask)
    rect = cv2.boundingRect(non_zero_coords)
    
    # Adjust the rectangle coordinates and dimensions to include the margin
    margin = 5
    x, y, w, h = rect
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(circle_mask.shape[1]-x, w+2*margin)
    h = min(circle_mask.shape[0]-y, h+2*margin)

    # Draw the rectangle
    # cv2.rectangle(circle_mask, (x, y), (x+w, y+h), (1), 2)
    print(x, y, x+w, y+h)
    
    # repeate circle mask for all slices
    circle_mask = np.repeat(circle_mask[:, :, np.newaxis], cardio_data.shape[2], axis=2)
    print(circle_mask.shape)

    # Save the new NIfTI image
    # nib.save(new_nifti, f'{output_path}/cardio_circle_mask.nii.gz')
    new_nifti = nib.Nifti1Image(circle_mask, output_img.affine)
    nib.save(new_nifti, f'{output_path}/cardio_circle_mask.nii.gz')
    
    ct_data = input_img.get_fdata()
    min_value = ct_data.min()
    ct_data = ct_data * circle_mask
    ct_data[circle_mask == 0] = min_value
    print(ct_data.max())
    
    ct_data = ct_data[y:y+h, x:x+w]
    print(ct_data.shape)
    
    
    # segmentations = nib.load(os.path.join(output_path, "segs2", "segmentations_test.nii.gz"))
    
    # # Create a new NIfTI image from the modified data
    new_nifti = nib.Nifti1Image(ct_data, output_img.affine)

    # Save the new NIfTI image
    # nib.save(new_nifti, f'{output_path}/cardio_circle_mask.nii.gz')
    nib.save(new_nifti, f'{output_path}/cardio_fake_gated.nii.gz')