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
    root_path = 'data/EXAMES/Exames_NIFTI'
    pacients = os.listdir(root_path)
    
    # heart_segs = nib.load('data/EXAMES/Exames_NIFTI')
    exam_type = 'partes_moles'
    for pacient in tqdm(pacients):
        print(pacient)
        # pacient = '183077'
        pacient_path = os.path.join(root_path, pacient, pacient)
        # heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{pacient}/{pacient}/partes_moles_HeartSegs.nii.gz')
        heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{pacient}/{pacient}/partes_moles_HeartSegs.nii.gz')
        
        heart_mask = heart_segs_data.get_fdata()
        heart_mask[heart_mask != 0] = 1
        output_path = f'data/EXAMES/Exames_NIFTI/{pacient}/{pacient}'
        
        # dilation of the mask
        kernel = np.ones((7,7), np.uint8)
        print(heart_mask.shape)
        cardio_data = cv2.dilate(heart_mask, kernel, iterations=3)

        print(cardio_data.shape)
        1/0
        new_nifti = nib.Nifti1Image(cardio_data, heart_segs_data.affine)
        nib.save(new_nifti, f'{output_path}/cardiac_IncreasedMask.nii.gz')

        nifti_files = os.listdir(pacient_path)
        exclude_files = ['binary_lesion', 'multi_label', 'multi_lesion']
        nifti_files = [
            file 
            for file in nifti_files 
            if not any(f in file for f in exclude_files)
        ]
        try:
            # motion_filename = [file for file in nifti_files if 'partes_moles_body' in file][0]
            motion_filename = [file for file in nifti_files if 'cardiac' in file][0]
        except:
            print(f'cardiac not found!: {pacient}')
            break
        
        input_img = nib.load(os.path.join(pacient_path, motion_filename))#.get_fdata()
        ct_data = input_img.get_fdata()
        
        calc_candidates = ct_data.copy()
        calc_candidates[calc_candidates < 130] = 0
        calc_candidates[calc_candidates >= 130] = 1
        
        # Create and save a new NIfTI image from the modified data
        new_nifti = nib.Nifti1Image(calc_candidates, input_img.affine)
        nib.save(new_nifti, f'{output_path}/partes_moles_CalciumCandidates_Mask.nii.gz')