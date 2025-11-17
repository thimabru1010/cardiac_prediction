import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import json
from utils import get_basename, load_nifti_sitk, create_save_nifti
import SimpleITK as sitk
from time import time

def extract_ids_mask(mask, ids):
    mask_tmp = np.zeros_like(mask)
    min_id = min(ids)
    for id in ids:
        mask_tmp[mask == id] = min_id
    return mask_tmp

def dilate_3d(arr, kernel, iters):
    out = np.zeros_like(arr)
    for z in range(arr.shape[2]):
        out[:,:,z] = cv2.dilate(arr[:,:,z].astype(np.uint8), kernel, iterations=iters)
    return out
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment Heart and Bones from NIfTI files')
    parser.add_argument('--root_path', type=str, default='data/ExamesArya_NIFTI2', help='Root path to the NIfTI files')
    parser.add_argument('--output_path', type=str, default='data/ExamesArya_NIFTI2', help='Output path for the segmented images')
    args = parser.parse_args()

    # root_path = args.root_path
    patients = os.listdir(args.root_path)

    # Load json file with the TotalSeg classes
    json_path = 'TotalSeg_classes.json'
    # Load the JSON file
    with open(json_path, 'r') as file:
        totalseg_classes = json.load(file)

    # Print the loaded TotalSeg classes
    # print(totalseg_classes)
    
    cardio_ids = [51]
    ribs_ids = list(range(92, 116))
    vertebra_ids = list(range(26, 50))
    esternum_ids = [116, 117]
    
    cardio_classes = [totalseg_classes[str(id)] for id in cardio_ids]
    ribs_classes = [totalseg_classes[str(id)] for id in ribs_ids]
    vertebra_classes = [totalseg_classes[str(id)] for id in vertebra_ids]
    esternum_classes = [totalseg_classes[str(id)] for id in esternum_ids]
    rois = cardio_classes + ribs_classes + vertebra_classes + esternum_classes
    
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
    keywords = ['non_gated']  
    
    dilation_kernel = np.ones((3,3), np.uint8)
    heart_dilation_kernel = np.ones((10,10), np.uint8)  
        
    for patient in tqdm(patients):
        print(patient)
        # patient = '105655'
        patient_path = os.path.join(args.root_path, patient)
        nifti_files = os.listdir(patient_path)
        motion_filename = get_basename(nifti_files, exclude_files=exclude_files, keywords=keywords)
        print(f"Processing {motion_filename} - Patient {patient}")

        # Needs to use nibabel for total segmentator
        input_img = nib.load(os.path.join(patient_path, motion_filename))
        # print(input_img)
        # input_img, input_array = load_nifti_sitk(os.path.join(patient_path, motion_filename), return_numpy=True)
        start = time()
        output_img = totalsegmentator(
            input_img,
            task='total',
            roi_subset=rois,
            device='gpu')
        end = time()
        print(f'TotalSegmentator inference time: {end - start:.2f} seconds')
        output_array = output_img.get_fdata()
        
        print(np.unique(output_array, return_counts=True))
        # output_array = sitk.GetArrayFromImage(output_img)
        heart_array = extract_ids_mask(output_array, cardio_ids)
        output_array[output_array == 51] = 0
        ribs_array = extract_ids_mask(output_array, ribs_ids)
        vertebra_array = extract_ids_mask(output_array, vertebra_ids)
        esternum_array = extract_ids_mask(output_array, esternum_ids)

        # Dilate masks
        print(heart_array.shape, heart_array.min(), heart_array.max(), heart_array.sum())
        print(ribs_array.shape, ribs_array.min(), ribs_array.max(), ribs_array.sum())
        print(vertebra_array.shape, vertebra_array.min(), vertebra_array.max(), vertebra_array.sum())
        print(esternum_array.shape, esternum_array.min(), esternum_array.max(), esternum_array.sum())
        heart_array = dilate_3d(heart_array, dilation_kernel, 5)
        ribs_array = dilate_3d(ribs_array, dilation_kernel, 3)
        vertebra_array = dilate_3d(vertebra_array, dilation_kernel, 2)
        esternum_array = dilate_3d(esternum_array, dilation_kernel, 2)

        bones_array = ribs_array + vertebra_array + esternum_array

        heart_big_array = dilate_3d(heart_array, heart_dilation_kernel, 3)

        patient_output_path = os.path.join(args.output_path, patient)
        os.makedirs(patient_output_path, exist_ok=True)
        create_save_nifti(heart_array, output_img.affine, os.path.join(patient_output_path, 'non_gated_HeartSegs.nii.gz'))
        create_save_nifti(bones_array, output_img.affine, os.path.join(patient_output_path, 'non_gated_BonesSegs.nii.gz'))
        create_save_nifti(heart_big_array, output_img.affine, os.path.join(patient_output_path, 'non_gated_HeartSegs_dilat_k=10.nii.gz'))
        
        # sitk.WriteImage(heart_img, os.path.join(patient_output_path, 'non_gated_HeartSegs.nii.gz'))
        # sitk.WriteImage(bones_img, os.path.join(patient_output_path, 'non_gated_BonesSegs.nii.gz'))
        # sitk.WriteImage(heart_big_img, os.path.join(patient_output_path, 'non_gated_HeartSegs_dilat_k=10.nii.gz'))

        print('Saved:', os.path.join(patient_output_path, 'non_gated_HeartSegs.nii.gz'))
        print('Saved:', os.path.join(patient_output_path, 'non_gated_BonesSegs.nii.gz'))
        print('Saved:', os.path.join(patient_output_path, 'non_gated_HeartSegs_dilat_k=10.nii.gz'))


    print('Segmentation finished!')

