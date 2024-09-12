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
import SimpleITK as sitk
import pandas as pd

def get_basename(files):
    # Exclude the multi_label, multi_lesion and binary_lesion files with these names inside the list
    exclusion_names=['multi_label', 'multi_lesion', 'binary_lesion']
    gated_exam_basename = [file for file in files if 'cardiac' in file]
    exclusion_files = [file for file in gated_exam_basename if any(f in file for f in exclusion_names)]
    gated_exam_basename.remove(exclusion_files[0])
    gated_exam_basename.remove(exclusion_files[1])
    gated_exam_basename.remove(exclusion_files[2])
    return gated_exam_basename[0]

def calculate_direct_score(exam, mask, pixel_spacing):
    # Dilate the mask
    # kernel = np.ones((10,10), np.uint8)
    # gated_mask = cv2.dilate(gated_mask, kernel, iterations=3)
    mask[mask != 0] = 1
    exam_calcification = exam * mask
    slice_thickness = pixel_spacing[2]
    return exam_calcification.sum() * pixel_spacing[0] * pixel_spacing[1] * slice_thickness

def calculate_estimated_score(exam, mask, pixel_spacing):
    # Dilate the mask
    # kernel = np.ones((10,10), np.uint8)
    # gated_mask = cv2.dilate(gated_mask, kernel, iterations=3)
    mask[mask != 0] = 1
    exam_calcification = exam * mask
    calcified_candidates = exam_calcification[exam_calcification >= 130]
    slice_thickness = pixel_spacing[2]
    return calcified_candidates.sum() * pixel_spacing[0] * pixel_spacing[1] * slice_thickness

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    
    save_path = 'data/EXAMES/Exames_Separados/test_segs'
        
    pacients = os.listdir(root_path)
    results = []
    for pacient in pacients:
        print(pacient)
        # print('Fake Gated Agaston Score Calculation')
        fg_exam_path = f'{root_path}/{pacient}/{pacient}/partes_moles_FakeGated.nii.gz'
        # Region of Intrest of the calcified candidates
        fg_mask_lab_path = f'{root_path}/{pacient}/{pacient}/partes_moles_FakeGated.nii_multi_label.nii.gz'
        # Exacly Segmentation of the calcified regions
        fg_mask_les_path = f'{root_path}/{pacient}/{pacient}/partes_moles_FakeGated.nii_multi_lesion.nii.gz'
        
        fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
        fg_mask_lab_img = nib.load(fg_mask_lab_path)#.get_fdata()
        fg_mask_les_img = nib.load(fg_mask_les_path)#.get_fdata()
        
        fg_exam = fg_exam_img.get_fdata()
        fg_mask_lab = fg_mask_lab_img.get_fdata().transpose(1, 2, 0)
        fg_mask_les = fg_mask_les_img.get_fdata().transpose(1, 2, 0)
        
        pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
        
        estimated_score_fg = calculate_estimated_score(fg_exam, fg_mask_lab, pixel_spacing)
        direct_score_fg = calculate_direct_score(fg_exam, fg_mask_les, pixel_spacing)
        
        # print('Estimated Faked Gated score:', estimated_score_fg)
        # print('Direct Fake Gated Score:', direct_score_fg)
        
        # print('Gated Agaston Score Calculation')
        gated_exam_basename = get_basename(os.listdir(f'{root_path}/{pacient}/{pacient}'))
        # print(gated_exam_basename)
        gated_exam_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename}'
        gated_mask_les_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename[:-3]}_multi_lesion.nii.gz'
        gated_mask_lab_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename[:-3]}_multi_label.nii.gz'
        
        gated_exam_img = nib.load(gated_exam_path)#.get_fdata()
        gated_mask_les_img = nib.load(gated_mask_les_path)#.get_fdata()
        gated_mask_lab_img = nib.load(gated_mask_lab_path)#.get_fdata()
        
        gated_exam = gated_exam_img.get_fdata()
        gated_mask_les = gated_mask_les_img.get_fdata().transpose(1, 2, 0)
        gated_mask_lab = gated_mask_lab_img.get_fdata().transpose(1, 2, 0)
        
        pixel_spacing = gated_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
        
        estimated_score_gated = calculate_estimated_score(gated_exam, gated_mask_lab, pixel_spacing)
        direct_score_gated = calculate_direct_score(gated_exam, gated_mask_les, pixel_spacing)

        # Results
        print('Estimated Score gated:', estimated_score_gated)
        print('Direct Score gated:', direct_score_gated)
        
        print('Estimated Faked Gated:', estimated_score_fg)
        print('Direct Fake Gated:', direct_score_fg)
        
        results.append([pacient, estimated_score_gated, direct_score_gated, estimated_score_fg, direct_score_fg])

    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    print(df_score_ref.head())
    df_score_ref['Pacient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    df = pd.DataFrame(results, columns=['Pacient', 'Estimated Score Gated', 'Direct Score Gated', 'Estimated Score Fake Gated', 'Direct Score Fake Gated'])
    df['Pacient'] = df['Pacient'].astype(int)
    df[['Estimated Score Gated', 'Direct Score Gated', 'Estimated Score Fake Gated', 'Direct Score Fake Gated']] = df[['Estimated Score Gated', 'Direct Score Gated', 'Estimated Score Fake Gated', 'Direct Score Fake Gated']].astype(int)
    df = pd.merge(df, df_score_ref[['Pacient', 'Escore']], on='Pacient', how='left')
    
    df.to_csv('data/EXAMES/calcium_score_estimations.csv', index=False)

    
    
    
    
    
    