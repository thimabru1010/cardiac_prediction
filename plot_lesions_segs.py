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
import argparse
from utils import get_basename, set_string_parameters, calculate_area
        
# Create main
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Calculate Agaston Score')
    argparser.add_argument('--root_path', type=str, default='data/EXAMES', help='Root path of the exams')
    argparser.add_argument('--dilate', action='store_true', help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_kernel', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_it', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--partes_moles', action='store_true', help='Whether to infer partes_moles exams')
    argparser.add_argument('--avg', type=int, default=4, help='Number of slices to average')
    argparser.add_argument('--cac_th', type=int, default=130, help='Calcification Threshold in HU')
    args = argparser.parse_args()

    partes_moles_basename, avg_str = set_string_parameters(args.avg)
            
    # Load Calcium Score
    cac_scores_path = os.path.join(args.root_path, f'Calcium_Scores_Estimations/Fake_Gated/avg={args.avg}/{args.cac_th}', 'calcium_score_estimations_avg=4.csv')
    cac_scores = pd.read_csv(cac_scores_path)
    # data/EXAMES/Calcium_Scores_Estimations/Fake_Gated/avg=4/130/calcium_score_estimations_avg=4.csv
    
    high_score_patients = ['74657', '176253', '182447']
    low_score_patients = ['176063', '80376']
    
    patients = high_score_patients + low_score_patients
    patients = os.listdir(f'{args.root_path}/Exames_NIFTI')
    for patient in patients:
        
        fg_exam_path = f'{args.root_path}/Exames_NIFTI/{patient}/{patient}/{partes_moles_basename}.nii.gz'
        fg_mask_les_path = f'{args.root_path}/Exames_NIFTI/{patient}/{patient}/{partes_moles_basename}_multi_lesion.nii.gz'
        
        fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
        fg_mask_les_img = nib.load(fg_mask_les_path)#.get_fdata()
        
        print(fg_exam_img.spac)
        
        fg_exam = fg_exam_img.get_fdata()
        fg_mask_les = fg_mask_les_img.get_fdata()#.transpose(1, 2, 0)
        
        # Find the slice with the biggest lesion area
        slice_areas = calculate_area(fg_mask_les, axis=(0, 1))
        slice_areas_idx = np.argsort(slice_areas)
        print(slice_areas)
        max_slice = slice_areas_idx[-1]
        print(max_slice)
        
        
        break
        
        
        
    
    
    
    