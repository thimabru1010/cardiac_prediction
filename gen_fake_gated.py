import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from skimage import restoration, io, color
from skimage.measure import blur_effect
import pandas as pd
from utils import get_basename, create_save_nifti
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import zoom

def circumscribing_rectangle(center, radius):
    x_center, y_center = center
    # diameter = 2 * radius
    
    # Coordinates of the rectangle
    x_min = x_center - radius
    x_max = x_center + radius
    y_min = y_center - radius
    y_max = y_center + radius
    
    # Return x, y, w, h
    return x_min, y_min, x_max - x_min, y_max - y_min

def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated_CircleMask', 'multi_label', 'multi_lesion', 'binary_lesion']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

def upsample_fill_mask(mask, new_shape):
    mask2 = np.zeros((new_shape[0], new_shape[1], mask.shape[2]))
    for i in range(mask.shape[2]):
        mask2[:, :, i] = cv2.resize(mask[:, :, i], new_shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        mask2[:, :, i] = binary_fill_holes(mask2[:, :, i])
    return mask2

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    patients = os.listdir(root_path)
    
    # heart_segs = nib.load('data/EXAMES/Exames_NIFTI')
    exam_type = 'partes_moles'
    new_img_size = (512, 512)
    blur_metrics = []
    dim_scale_factors = (1, 1, 1/4)
    avg = 0
    for patient in tqdm(patients):
        patient = '176063'
        print(patient)
        
        # if patient != '176064':
        #     continue
        
        # if patient == '176064':
        #     print('Here')
            
        output_path = f'data/EXAMES/Exames_NIFTI/{patient}/{patient}'
        patient_path = os.path.join(root_path, patient, patient)
        # heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_HeartSegs.nii.gz')
        heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_HeartSegs.nii.gz')
        ribs_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_RibsSegs.nii.gz')
        vertebral_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_VertebraSegs.nii.gz')
        
        heart_mask = heart_segs_data.get_fdata()
        ribs_mask = ribs_segs_data.get_fdata()
        vertebral_mask = vertebral_segs_data.get_fdata()
        
        if heart_mask[heart_mask != 0].shape[0] == 0:
            print(f'No heart mask found in {patient}')
            continue
        heart_mask[heart_mask != 0] = 1
        
        # dilation of the mask to create one single big mask and estimate better the centroid
        kernel = np.ones((10,10), np.uint8)
        cardio_data = cv2.dilate(heart_mask, kernel, iterations=3)
        
        # Get the slice with the maximum area
        count_area = np.sum(cardio_data, axis=(0, 1))
        max_index = np.argmax(count_area)
        max_slice = cardio_data[:, :, max_index]
        coordinates = np.argwhere(max_slice)

        # Calculate the centroid from the Total Segmentator Heart Mask
        centroid = coordinates.mean(axis=0)
        circle_mask = np.zeros(max_slice.shape)
        radius = 150
        center = (int(centroid[1]), int(centroid[0]))
        cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
        # Get the rectangle circunscribing the circle
        # Get the coordinates of the non-zero pixels in the circle mask
        non_zero_coords = np.argwhere(circle_mask)
        rect = cv2.boundingRect(non_zero_coords)
        rect = circumscribing_rectangle(center, radius)
        x, y, w, h = rect
        
        # repeate circle mask for all slices
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], cardio_data.shape[2], axis=2)

        # Save the new NIfTI image
        create_save_nifti(circle_mask, heart_segs_data.affine, f'{output_path}/partes_moles_FakeGated_CircleMask.nii.gz')
        
        nifti_files = os.listdir(patient_path)
        motion_filename = get_partes_moles_basename(nifti_files)
        input_img = nib.load(os.path.join(patient_path, motion_filename))#.get_fdata()
        ct_data = input_img.get_fdata()
        print('Original shape:', ct_data.shape)
        
        min_value = ct_data.min()
        ct_data = ct_data * circle_mask
        ct_data[circle_mask == 0] = min_value
        # print(ct_data.max())
        
        ct_data = ct_data[y:y+h, x:x+w]
        cardio_data = cardio_data[y:y+h, x:x+w]
        ribs_mask = ribs_mask[y:y+h, x:x+w]
        vertebral_mask = vertebral_mask[y:y+h, x:x+w]
        
        cardio_data_upsampled = upsample_fill_mask(cardio_data, new_img_size)
        ribs_mask_upsampled = upsample_fill_mask(ribs_mask, new_img_size)
        vertebral_mask_upsampled = upsample_fill_mask(vertebral_mask, new_img_size)
            
        create_save_nifti(cardio_data_upsampled, heart_segs_data.affine, f'{output_path}/partes_moles_HeartSegs_FakeGated.nii.gz')
        create_save_nifti(ribs_mask_upsampled, ribs_segs_data.affine, f'{output_path}/partes_moles_RibsSegs_FakeGated.nii.gz')
        create_save_nifti(vertebral_mask_upsampled, vertebral_segs_data.affine, f'{output_path}/partes_moles_VertebralSegs_FakeGated.nii.gz')
        
        print(f'{output_path}/partes_moles_HeartSegs_FakeGated.nii.gz')
        print(f'{output_path}/partes_moles_RibsSegs_FakeGated.nii.gz')
        print(f'{output_path}/partes_moles_VertebralSegs_FakeGated.nii.gz')
        
        # Reduce the size of channels the image by interpolation
        ct_data = zoom(ct_data, dim_scale_factors, order=3)
        cardio_data = zoom(cardio_data, dim_scale_factors, order=0, mode='nearest')
        ribs_mask = zoom(ribs_mask, dim_scale_factors, order=0, mode='nearest')
        vertebral_mask = zoom(vertebral_mask, dim_scale_factors, order=0, mode='nearest')
        
        print('Reduced shape Exam:', ct_data.shape)
        print('Reduced shape Mask:', cardio_data.shape)
        print('Reduced shape Ribs:', ribs_mask.shape)
        print('Reduced shape Vertebra:', vertebral_mask.shape)
        
        cardio_data[cardio_data > 0] = 1
        ribs_mask[ribs_mask > 0] = 1
        vertebral_mask[vertebral_mask > 0] = 1
        
        ct_data_upsampled = np.zeros((new_img_size[0], new_img_size[1], ct_data.shape[2]))
        for i in range(ct_data.shape[2]):
            ct_data_upsampled[:, :, i] = cv2.resize(ct_data[:, :, i], new_img_size, interpolation=cv2.INTER_LANCZOS4)
            
        cardio_data = upsample_fill_mask(cardio_data, new_img_size)
        ribs_mask = upsample_fill_mask(ribs_mask, new_img_size)
        vertebral_mask = upsample_fill_mask(vertebral_mask, new_img_size)
        
        # Create a new NIfTI image from the modified data
        create_save_nifti(ct_data_upsampled, heart_segs_data.affine, f'{output_path}/partes_moles_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(cardio_data, heart_segs_data.affine, f'{output_path}/partes_moles_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(ribs_mask, ribs_segs_data.affine, f'{output_path}/partes_moles_RibsSegs_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(vertebral_mask, vertebral_segs_data.affine, f'{output_path}/partes_moles_VertebraSegs_FakeGated_avg_slices=4.nii.gz')
        
        print(f'{output_path}/partes_moles_FakeGated_avg_slices=4.nii.gz')
        print(f'{output_path}/partes_moles_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        print(f'{output_path}/partes_moles_RibsSegs_FakeGated_avg_slices=4.nii.gz')
        print(f'{output_path}/partes_moles_VertebraSegs_FakeGated_avg_slices=4.nii.gz')
        1/0