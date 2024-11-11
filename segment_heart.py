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

def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

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
    
    cardio_ids = list(range(51, 68))
    
    cardio_classes = [totalseg_classes[str(id)] for id in cardio_ids]
    
    for patient in tqdm(patients):
        print(patient)
        patient_path = os.path.join(root_path, patient, patient)
        nifti_files = os.listdir(patient_path)
        motion_filename = get_partes_moles_basename(nifti_files)
        print(f"Processing {motion_filename} - Patient {patient}")
        # try:
        #     # motion_filename = [file for file in nifti_files if 'partes_moles_body' in file][0]
        #     motion_filename = get_partes_moles_basename(nifti_files)
        #     print(f"Processing {motion_filename}")
        # except:
        #     print(f'partes_moles_body not found!: {patient}')
        #     break
        
        # print(motion_filename)
        # 1/0
        # infer exams
        # output_path = os.path.join('data/EXAMES/Exames_NIFTI_HeartSegs', patient, patient)
        output_path = os.path.join('data/EXAMES/Exames_NIFTI', patient, patient)
        os.makedirs(output_path, exist_ok=True)
        
        #for filename in [gated_filename, motion_filename]:
        input_img = nib.load(os.path.join(patient_path, motion_filename))
        output_img = totalsegmentator(input_img, task='total', roi_subset=cardio_classes)
        
        # basename = os.path.splitext(motion_filename)[0]
        # nib.save(output_img, f'{output_path}/cardiac_HeartSegs.nii.gz')
        # print('Saved:', f'{output_path}/cardiac_HeartSegs.nii.gz')
        
        nib.save(output_img, f'{output_path}/partes_moles_HeartSegs.nii.gz')
        print('Saved:', f'{output_path}/partes_moles_HeartSegs.nii.gz')

    print('Segmentation finished!')

