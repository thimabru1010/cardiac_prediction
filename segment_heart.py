import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti
import json
from utils import get_basename, create_save_nifti

def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

def extract_ids_mask(mask, ids):
    mask_tmp = np.zeros_like(mask)
    for id in ids:
        mask_tmp[mask == id] = id
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
    
    cardio_classes = [totalseg_classes[str(id)] for id in cardio_ids]
    ribs_classes = [totalseg_classes[str(id)] for id in ribs_ids]
    vertebra_classes = [totalseg_classes[str(id)] for id in vertebra_ids]
    rois = cardio_classes + ribs_classes + vertebra_classes
    
    exclude_files = ['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask', 'multi_label', 'multi_lesion', 'binary_lesion']
    keywords = ['partes_moles_body', 'mediastino']  
    
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
        ribs_img = extract_ids_mask(img, ribs_ids)
        vertebra_img = extract_ids_mask(img, vertebra_ids)

        create_save_nifti(heart_img, output_img.affine, f'{output_path}/partes_moles_HeartSegs.nii.gz')
        create_save_nifti(ribs_img, output_img.affine, f'{output_path}/partes_moles_RibsSegs.nii.gz')
        create_save_nifti(vertebra_img, output_img.affine, f'{output_path}/partes_moles_VertebraSegs.nii.gz')
        
        print('Saved:', f'{output_path}/partes_moles_HeartSegs.nii.gz')
        print('Saved:', f'{output_path}/partes_moles_RibsSegs.nii.gz')
        print('Saved:', f'{output_path}/partes_moles_VertebraSegs.nii.gz')
        
        

    print('Segmentation finished!')

