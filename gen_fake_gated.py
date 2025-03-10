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

def upsample_mask(mask, new_shape):
    mask2 = np.zeros((new_shape[0], new_shape[1], mask.shape[2]))
    for i in range(mask.shape[2]):
        mask2[:, :, i] = cv2.resize(mask[:, :, i], new_shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return mask2

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    patients = os.listdir(root_path)
    
    exam_type = 'partes_moles'
    new_img_size = (512, 512)
    blur_metrics = []
    dim_scale_factors = (1, 1, 1/4)
    avg = 0
    ribs_ids = list(range(92, 116))
    vertebra_ids = list(range(26, 50))
    esternum_ids = [116, 117]
    for patient in tqdm(patients):
        print(patient)
            
        output_path = f'data/EXAMES/Exames_NIFTI/{patient}/{patient}'
        patient_path = os.path.join(root_path, patient, patient)
        heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_HeartSegs.nii.gz')
        bones_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_BonesSegs.nii.gz')
        # heart_circle_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_HeartSegs_4Circle.nii.gz')
        
        heart_mask = heart_segs_data.get_fdata()
        bones_mask = bones_segs_data.get_fdata()
        # heart_circle_mask = heart_circle_segs_data.get_fdata()

        
        if heart_mask[heart_mask != 0].shape[0] == 0:
            print(f'No heart mask found in {patient}')
            continue
        heart_mask[heart_mask != 0] = 1
        heart_mask[heart_mask != 0] = 1
        
        # heart_mask = heart_mask.copy()
        # Get the slice with the maximum area
        count_area = np.sum(heart_mask, axis=(0, 1))
        max_index = np.argmax(count_area)
        max_slice = heart_mask[:, :, max_index]
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
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], heart_mask.shape[2], axis=2)

        # Save the new NIfTI image
        create_save_nifti(circle_mask, heart_segs_data.affine, f'{output_path}/partes_moles_FakeGated_CircleMask.nii.gz')
        
        nifti_files = os.listdir(patient_path)
        motion_filename = get_partes_moles_basename(nifti_files)
        print(f"Processing {motion_filename} - Patient {patient}")
        
        input_img = nib.load(os.path.join(patient_path, motion_filename))#.get_fdata()
        ct_data = input_img.get_fdata()
        print('Original shape:', ct_data.shape)
        
        min_value = ct_data.min()
        ct_data = ct_data * circle_mask
        ct_data[circle_mask == 0] = min_value
        
        ct_data = ct_data[y:y+h, x:x+w]
        heart_mask = heart_mask[y:y+h, x:x+w]
        bones_mask = bones_mask[y:y+h, x:x+w]
        
        # heart_mask_upsampled = upsample_fill_mask(heart_mask, new_img_size)
        ct_data_upsampled = np.zeros((new_img_size[0], new_img_size[1], ct_data.shape[2]))
        for i in range(ct_data.shape[2]):
            ct_data_upsampled[:, :, i] = cv2.resize(ct_data[:, :, i], new_img_size, interpolation=cv2.INTER_NEAREST)
            
        bones_mask_upsampled = upsample_mask(bones_mask, new_img_size)
        heart_mask_upsampled = upsample_mask(heart_mask, new_img_size)
        
        ct_data_no_bones = ct_data_upsampled.copy()
        ct_data_no_bones[bones_mask_upsampled > 0] = 0
        create_save_nifti(ct_data_no_bones, heart_segs_data.affine, f'{output_path}/partes_moles_FakeGated_no_bones.nii.gz')
        create_save_nifti(ct_data_upsampled, heart_segs_data.affine, f'{output_path}/partes_moles_FakeGated.nii.gz')
        create_save_nifti(bones_mask_upsampled, bones_segs_data.affine, f'{output_path}/partes_moles_BonesSegs_FakeGated.nii.gz')
        create_save_nifti(heart_mask_upsampled, heart_segs_data.affine, f'{output_path}/partes_moles_HeartSegs_FakeGated.nii.gz')
        
        print(f'{output_path}/partes_moles_HeartSegs_FakeGated.nii.gz')
        print(f'{output_path}/partes_moles_BonesSegs_FakeGated.nii.gz')
        print(f'{output_path}/partes_moles_FakeGated.nii.gz')
        
        new_z_spacing = 3.0
        imgSize = ct_data_upsampled.shape; # Tamanho do volume da ct
        imgNewSize = round(imgSize[2] / (new_z_spacing / input_img.header.get_zooms()[2])); # eu to pegando apenas o eixo Z
        dim_scale_factors = (1, 1, imgNewSize / imgSize[2])
        # data = imresize3(data, imgNewSize) %% Normalizando o voxel para 3mm apenas no eixo Z
        print(f'Scale factors: {dim_scale_factors}')

        # Reduce the size of channels the image by interpolation
        ct_data_upsampled = zoom(ct_data_upsampled, dim_scale_factors, order=5)
        heart_mask = zoom(heart_mask_upsampled, dim_scale_factors, order=0, mode='nearest')
        bones_mask = zoom(bones_mask_upsampled, dim_scale_factors, order=0, mode='nearest')
        
        print('Reduced shape Exam:', ct_data_upsampled.shape)
        print('Reduced shape Mask:', heart_mask.shape)
        print('Reduced shape Bones:', bones_mask.shape)
        
        bones_mask[bones_mask > 0] = 1
        
        # Create a new NIfTI image from the modified data
        # 7. Atualizar a matriz afim para refletir o novo espaçamento
        new_affine = input_img.affine.copy()
        new_affine[2, 2] *= (new_z_spacing / input_img.header.get_zooms()[2])  # Ajusta transformação no eixo Z

        create_save_nifti(ct_data_upsampled, new_affine, f'{output_path}/partes_moles_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(heart_mask, heart_segs_data.affine, f'{output_path}/partes_moles_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(bones_mask, new_affine, f'{output_path}/partes_moles_BonesSegs_FakeGated_avg_slices=4.nii.gz')
        
        print(f'{output_path}/partes_moles_FakeGated_avg_slices=4.nii.gz')
        print(f'{output_path}/partes_moles_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        print(f'{output_path}/partes_moles_BonesSegs_FakeGated_avg_slices=4.nii.gz')