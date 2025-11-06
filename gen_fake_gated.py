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
import argparse

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

def upsample_fill_mask(mask, new_shape):
    mask2 = np.zeros((new_shape[0], new_shape[1], mask.shape[2]))
    for i in range(mask.shape[2]):
        mask2[:, :, i] = cv2.resize(mask[:, :, i], new_shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        mask2[:, :, i] = binary_fill_holes(mask2[:, :, i])
    return mask2

def upsample_mask(mask, new_shape):
    mask2 = np.zeros((new_shape[0], new_shape[1], mask.shape[2]))
    for i in range(mask.shape[2]):
        mask2[:, :, i] = cv2.resize(mask[:, :, i], new_shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return mask2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Fake Gated Images')
    parser.add_argument('--root_path', type=str, default='data/ExamesArya_NIFTI2', help='Root path to the NIfTI files')
    parser.add_argument('--output_path', type=str, default='data/ExamesArya_NIFTI2', help='Output path for the generated images')
    args = parser.parse_args()

    root_path = args.root_path
    output_path = args.output_path
    patients = os.listdir(root_path)
    
    exclude_files = ['_HeartSegs',
                     '_BonesSegs',
                     '_FakeGated',
                     '_FakeGated_CircleMask',
                     'multi_label',
                     'multi_lesion',
                     'binary_lesion',
                     '_mask',
                     'CalciumCandidates',
                     'IncreasedLesion',
                     'LesionsSingleLesions',
                     'SingleLesions']
        
    exam_type = 'non_gated'
    new_img_size = (512, 512)
    blur_metrics = []
    dim_scale_factors = (1, 1, 1/4)
    avg = 0
    ribs_ids = list(range(92, 116))
    vertebra_ids = list(range(26, 50))
    esternum_ids = [116, 117]
    for patient in tqdm(patients):
        patient = '105655'
        print(patient)
        patient_output_path = os.path.join(output_path, patient)
        patient_path = os.path.join(root_path, patient)
        heart_segs_data = nib.load(os.path.join(patient_path, 'non_gated_HeartSegs.nii.gz'))
        bones_segs_data = nib.load(os.path.join(patient_path, 'non_gated_BonesSegs.nii.gz'))
        heart_circle_segs_data = nib.load(os.path.join(patient_path, 'non_gated_HeartSegs_dilat_k=10.nii.gz'))

        heart_mask = heart_segs_data.get_fdata()
        bones_mask = bones_segs_data.get_fdata()
        heart_circle_mask = heart_circle_segs_data.get_fdata()
        
        if heart_mask[heart_mask != 0].shape[0] == 0:
            print(f'No heart mask found in {patient}')
            continue
        heart_mask[heart_mask != 0] = 1
        heart_circle_mask[heart_circle_mask != 0] = 1
        
        # Get the slice with the maximum area
        count_area = np.sum(heart_circle_mask, axis=(0, 1))
        max_index = np.argmax(count_area)
        max_slice = heart_circle_mask[:, :, max_index]
        coordinates = np.argwhere(max_slice)

        # Creating the zoom circle simulating gated exam
        # Calculate the centroid from the Total Segmentator Heart Mask
        centroid = coordinates.mean(axis=0)
        circle_mask = np.zeros(max_slice.shape)
        radius = 150
        center = (int(centroid[1]), int(centroid[0]))
        cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
        # Get the rectangle circunscribing the circle
        rect = circumscribing_rectangle(center, radius)
        x, y, w, h = rect
        
        # repeate circle mask for all slices
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], heart_circle_mask.shape[2], axis=2)

        # Save the new NIfTI image
        create_save_nifti(circle_mask, heart_segs_data.affine, f'{patient_output_path}/non_gated_FakeGated_CircleMask.nii.gz')
        
        nifti_files = os.listdir(patient_path)
        motion_filename = get_basename(nifti_files, exclude_files=exclude_files, keywords=['non_gated'])
        print(f"Processing {motion_filename} - Patient {patient}")
        
        input_img = nib.load(os.path.join(patient_path, motion_filename))#.get_fdata()
        ct_data = input_img.get_fdata()
        print('Original shape:', ct_data.shape)
        print(f"Original CT values: min={ct_data.min()}, max={ct_data.max()}, mean={ct_data.mean()}")
        
        min_value = ct_data.min()
        ct_data = ct_data * circle_mask
        ct_data[circle_mask == 0] = min_value
        
        ct_data = ct_data[y:y+h, x:x+w]
        heart_mask = heart_mask[y:y+h, x:x+w]
        bones_mask = bones_mask[y:y+h, x:x+w]
        
        print(f"Cropped CT values: min={ct_data.min()}, max={ct_data.max()}, mean={ct_data.mean()}")
        # heart_mask_upsampled = upsample_fill_mask(heart_mask, new_img_size)
        ct_data_upsampled = np.zeros((new_img_size[0], new_img_size[1], ct_data.shape[2]))
        for i in range(ct_data.shape[2]):
            ct_data_upsampled[:, :, i] = cv2.resize(ct_data[:, :, i], new_img_size, interpolation=cv2.INTER_NEAREST)
        
        print(f"Upsampled CT values: min={ct_data_upsampled.min()}, max={ct_data_upsampled.max()}, mean={ct_data_upsampled.mean()}")
        bones_mask_upsampled = upsample_mask(bones_mask, new_img_size)
        heart_mask_upsampled = upsample_mask(heart_mask, new_img_size)
        
        ct_data_no_bones = ct_data_upsampled.copy()
        ct_data_no_bones[bones_mask_upsampled > 0] = 0
        create_save_nifti(ct_data_no_bones, heart_segs_data.affine, f'{patient_output_path}/non_gated_FakeGated_no_bones.nii.gz')
        create_save_nifti(ct_data_upsampled, heart_segs_data.affine, f'{patient_output_path}/non_gated_FakeGated.nii.gz')
        create_save_nifti(bones_mask_upsampled, bones_segs_data.affine, f'{patient_output_path}/non_gated_BonesSegs_FakeGated.nii.gz')
        create_save_nifti(heart_mask_upsampled, heart_segs_data.affine, f'{patient_output_path}/non_gated_HeartSegs_FakeGated.nii.gz')

        print(f'{patient_output_path}/non_gated_HeartSegs_FakeGated.nii.gz')
        print(f'{patient_output_path}/non_gated_BonesSegs_FakeGated.nii.gz')
        print(f'{patient_output_path}/non_gated_FakeGated.nii.gz')

        new_z_spacing = 3.0 # Fixed value of gated exams
        imgSize = ct_data_upsampled.shape; # Tamanho do volume da ct
        imgNewSize = round(imgSize[2] / (new_z_spacing / input_img.header.get_zooms()[2])); # eu to pegando apenas o eixo Z
        dim_scale_factors = (1, 1, imgNewSize / imgSize[2])
        # data = imresize3(data, imgNewSize) %% Normalizando o voxel para 3mm apenas no eixo Z
        print(f'Scale factors: {dim_scale_factors}')

        # Reduce the size of channels the image by interpolation
        ct_data_upsampled = zoom(ct_data_upsampled, dim_scale_factors, order=3)
        heart_mask = zoom(heart_mask_upsampled, dim_scale_factors, order=0, mode='nearest')
        bones_mask = zoom(bones_mask_upsampled, dim_scale_factors, order=0, mode='nearest')
        
        print(f"After zoom CT values: min={ct_data_upsampled.min()}, max={ct_data_upsampled.max()}, mean={ct_data_upsampled.mean()}")
        print('Reduced shape Exam:', ct_data_upsampled.shape)
        print('Reduced shape Mask:', heart_mask.shape)
        print('Reduced shape Bones:', bones_mask.shape)
        
        bones_mask[bones_mask > 0] = 1
        
        # Create a new NIfTI image from the modified data
        # 7. Atualizar a matriz afim para refletir o novo espaçamento
        new_affine = input_img.affine.copy()
        new_affine[2, 2] *= (new_z_spacing / input_img.header.get_zooms()[2])  # Ajusta transformação no eixo Z

        create_save_nifti(ct_data_upsampled, new_affine, f'{patient_output_path}/non_gated_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(heart_mask, new_affine, f'{patient_output_path}/non_gated_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(bones_mask, new_affine, f'{patient_output_path}/non_gated_BonesSegs_FakeGated_avg_slices=4.nii.gz')

        print(f'{patient_output_path}/non_gated_FakeGated_avg_slices=4.nii.gz')
        print(f'{patient_output_path}/non_gated_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        print(f'{patient_output_path}/non_gated_BonesSegs_FakeGated_avg_slices=4.nii.gz')