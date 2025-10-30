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
import multiprocessing
from functools import partial

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

def process_patient(patient, root_path, output_path, exclude_files, new_img_size):
    try:
        print(f"Processing patient: {patient}")
        
        patient_output_path = os.path.join(output_path, patient)
        if not os.path.exists(patient_output_path):
            os.makedirs(patient_output_path)

        patient_path = os.path.join(root_path, patient)
        heart_segs_data = nib.load(os.path.join(patient_path, 'non_gated_HeartSegs.nii.gz'))
        bones_segs_data = nib.load(os.path.join(patient_path, 'non_gated_BonesSegs.nii.gz'))
        heart_circle_segs_data = nib.load(os.path.join(patient_path, 'non_gated_HeartSegs_dilat_k=10.nii.gz'))

        heart_mask = heart_segs_data.get_fdata()
        bones_mask = bones_segs_data.get_fdata()
        heart_circle_mask = heart_circle_segs_data.get_fdata()

        if heart_mask[heart_mask != 0].shape[0] == 0:
            print(f'No heart mask found in {patient}')
            return
        heart_mask[heart_mask != 0] = 1
        heart_circle_mask[heart_circle_mask != 0] = 1
        
        count_area = np.sum(heart_circle_mask, axis=(0, 1))
        max_index = np.argmax(count_area)
        max_slice = heart_circle_mask[:, :, max_index]
        coordinates = np.argwhere(max_slice)

        centroid = coordinates.mean(axis=0)
        circle_mask = np.zeros(max_slice.shape)
        radius = 150
        center = (int(centroid[1]), int(centroid[0]))
        cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
        
        non_zero_coords = np.argwhere(circle_mask)
        rect = cv2.boundingRect(non_zero_coords.astype(np.uint8))
        x, y, w, h = rect
        
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], heart_circle_mask.shape[2], axis=2)

        create_save_nifti(circle_mask, heart_segs_data.affine, f'{patient_output_path}/non_gated_FakeGated_CircleMask.nii.gz')
        
        nifti_files = os.listdir(patient_path)
        motion_filename = get_basename(nifti_files, exclude_files=exclude_files, keywords=['non_gated'])
        print(f"Processing {motion_filename} - Patient {patient}")
        
        input_img = nib.load(os.path.join(patient_path, motion_filename))
        ct_data = input_img.get_fdata()
        print(f'Original shape for {patient}:', ct_data.shape)
        
        min_value = ct_data.min()
        ct_data = ct_data * circle_mask
        ct_data[circle_mask == 0] = min_value
        
        ct_data = ct_data[y:y+h, x:x+w]
        heart_mask = heart_mask[y:y+h, x:x+w]
        bones_mask = bones_mask[y:y+h, x:x+w]
        
        ct_data_upsampled = np.zeros((new_img_size[0], new_img_size[1], ct_data.shape[2]))
        for i in range(ct_data.shape[2]):
            ct_data_upsampled[:, :, i] = cv2.resize(ct_data[:, :, i], new_img_size, interpolation=cv2.INTER_NEAREST)
            
        bones_mask_upsampled = upsample_mask(bones_mask, new_img_size)
        heart_mask_upsampled = upsample_mask(heart_mask, new_img_size)
        
        ct_data_no_bones = ct_data_upsampled.copy()
        ct_data_no_bones[bones_mask_upsampled > 0] = 0
        create_save_nifti(ct_data_no_bones, heart_segs_data.affine, f'{patient_output_path}/non_gated_FakeGated_no_bones.nii.gz')
        create_save_nifti(ct_data_upsampled, heart_segs_data.affine, f'{patient_output_path}/non_gated_FakeGated.nii.gz')
        create_save_nifti(bones_mask_upsampled, bones_segs_data.affine, f'{patient_output_path}/non_gated_BonesSegs_FakeGated.nii.gz')
        create_save_nifti(heart_mask_upsampled, heart_segs_data.affine, f'{patient_output_path}/non_gated_HeartSegs_FakeGated.nii.gz')

        new_z_spacing = 3.0
        imgSize = ct_data_upsampled.shape
        imgNewSize = round(imgSize[2] / (new_z_spacing / input_img.header.get_zooms()[2]))
        dim_scale_factors = (1, 1, imgNewSize / imgSize[2])
        
        ct_data_upsampled = zoom(ct_data_upsampled, dim_scale_factors, order=3)
        heart_mask = zoom(heart_mask_upsampled, dim_scale_factors, order=0, mode='nearest')
        bones_mask = zoom(bones_mask_upsampled, dim_scale_factors, order=0, mode='nearest')
        
        bones_mask[bones_mask > 0] = 1
        
        new_affine = input_img.affine.copy()
        new_affine[2, 2] *= (new_z_spacing / input_img.header.get_zooms()[2])

        create_save_nifti(ct_data_upsampled, new_affine, f'{patient_output_path}/non_gated_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(heart_mask, new_affine, f'{patient_output_path}/non_gated_HeartSegs_FakeGated_avg_slices=4.nii.gz')
        create_save_nifti(bones_mask, new_affine, f'{patient_output_path}/non_gated_BonesSegs_FakeGated_avg_slices=4.nii.gz')

        print(f"Finished processing patient: {patient}")
    except Exception as e:
        print(f"Error processing patient {patient}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Fake Gated Images')
    parser.add_argument('--root_path', type=str, default='data/ExamesArya_NIFTI2', help='Root path to the NIfTI files')
    parser.add_argument('--output_path', type=str, default='data/ExamesArya_NIFTI2_FakeGated', help='Output path for the generated images')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of parallel workers')
    args = parser.parse_args()

    root_path = args.root_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
        
    new_img_size = (512, 512)

    worker_func = partial(process_patient, root_path=root_path, output_path=output_path, exclude_files=exclude_files, new_img_size=new_img_size)

    with multiprocessing.Pool(args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, patients), total=len(patients)))

    print("All patients processed.")
