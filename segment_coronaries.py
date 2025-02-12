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

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    patients = os.listdir(root_path)
    
    exclude_files = ['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
    keywords_partes_moles = ['partes_moles_body', 'mediastino']
    
    partes_moles_basename = 'partes_moles_FakeGated_avg_slices=4'
    
    exclusion_patients = ['179238', '176064', '177222']
    for patient in tqdm(patients):
        print(patient)
        
        if patient in exclusion_patients:
            continue
        
        fg_exam_path = f'{root_path}/{patient}/{patient}/{partes_moles_basename}.nii.gz'
        fg_exam_img = nib.load(fg_exam_path)
        # fg_exam = fg_exam_img.get_fdata()
        
        output_img = totalsegmentator(fg_exam_img, task='coronary_arteries')
        
        # create_save_nifti(output_img, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_FakeGated_avg_slices=4_CoronaryArteries.nii.gz')
        nib.save(output_img, f'{root_path}/{patient}/{patient}/partes_moles_FakeGated_avg_slices=4_CoronaryArteries.nii.gz')