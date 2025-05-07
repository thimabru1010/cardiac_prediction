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
from utils import get_basename, create_save_nifti

def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

def extract_ids_mask(mask, ids):
    mask_tmp = np.zeros_like(mask)
    min_id = min(ids)
    for id in ids:
        mask_tmp[mask == id] = min_id
    return mask_tmp

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    patients = os.listdir(root_path)
    
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
    
    exclude_files = ['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask', 'multi_label', 'multi_lesion', 'binary_lesion']
    keywords = ['partes_moles_body', 'mediastino']  
    
    dilation_kernel = np.ones((3,3), np.uint8)
    heart_dilation_kernel = np.ones((10,10), np.uint8)  
        
    for patient in tqdm(patients):
        print(patient)
        patient_path = os.path.join(root_path, patient, patient)
        nifti_files = os.listdir(patient_path)
        motion_filename = get_partes_moles_basename(nifti_files)
        motion_filename = get_basename(nifti_files, exclude_files=exclude_files, keywords=keywords)
        # motion_filename = '4_partes_moles_body_10.nii.gz'
        print(f"Processing {motion_filename} - Patient {patient}")

        # output_path = os.path.join('data/EXAMES/Exames_NIFTI_HeartSegs', patient, patient)
        output_path = os.path.join('data/EXAMES/Exames_NIFTI', patient, patient)
        os.makedirs(output_path, exist_ok=True)
        
        #for filename in [gated_filename, motion_filename]:
        input_img = nib.load(os.path.join(patient_path, motion_filename))
        output_img = totalsegmentator(input_img, task='total', roi_subset=rois)
        img = output_img.get_fdata()
        heart_img = extract_ids_mask(img, cardio_ids)
        img[img == 51] = 0
        ribs_img = extract_ids_mask(img, ribs_ids)
        vertebra_img = extract_ids_mask(img, vertebra_ids)
        esternum_img = extract_ids_mask(img, esternum_ids)
        
        # Dilate masks
        heart_img = cv2.dilate(heart_img, dilation_kernel, iterations=5)
        ribs_img = cv2.dilate(ribs_img, dilation_kernel, iterations=3)
        vertebra_img = cv2.dilate(vertebra_img, dilation_kernel, iterations=2)
        esternum_img = cv2.dilate(esternum_img, dilation_kernel, iterations=2)
        
        bones_img = ribs_img + vertebra_img + esternum_img
        
        heart_big_img = cv2.dilate(heart_img, heart_dilation_kernel, iterations=3)

        create_save_nifti(heart_img, output_img.affine, f'{output_path}/partes_moles_HeartSegs.nii.gz')
        create_save_nifti(bones_img, output_img.affine, f'{output_path}/partes_moles_BonesSegs.nii.gz')
        create_save_nifti(heart_big_img, output_img.affine, f'{output_path}/partes_moles_HeartSegs_dilat_k=10.nii.gz')
        
        print('Saved:', f'{output_path}/partes_moles_HeartSegs.nii.gz')
        print('Saved:', f'{output_path}/partes_moles_BonesSegs.nii.gz')
        print('Saved:', f'{output_path}/partes_moles_HeartSegs_dilat_k=10.nii.gz')
        

    print('Segmentation finished!')

