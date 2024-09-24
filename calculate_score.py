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

def density_factor(max_HU):
    if max_HU >= 130 and max_HU < 200:
        return 1
    elif max_HU >= 200 and max_HU < 300:
        return 2
    elif max_HU >= 300 and max_HU < 400:
        return 3
    elif max_HU >= 400:
        return 4

def calculate_score(exam, mask, pixel_spacing, filter_calcium=True):
    agaston_score = 0
    max_cac = 0
    for channel in range(mask.shape[2]):
        _, lesions = cv2.connectedComponents(mask[:, :, channel].astype(np.uint8))
        lesion_labels = list(np.unique(lesions))
        # print(lesion_labels)
        if 0 in lesion_labels:
            lesion_labels.remove(0)
            
        for lesion_label in lesion_labels: # Exclude 0 (background)
            # num_lesion = 2
            lesion = lesions.copy()
            lesion[lesion != lesion_label] = 0
            lesion[lesion == lesion_label] = 1
            
            # if num_lesions >= 3:
            #     cv2.imwrite(f'data/EXAMES/lesion_{num_lesion}.png', lesion * 255)
            #     cv2.imwrite(f'data/EXAMES/mask_{num_lesion}.png', mask[:, :, channel] * 255)
            #     1/0
            
            # print(exam.shape, mask.shape)
            exam_calcification = exam[:, :, channel] * lesion
            calcified_candidates = exam_calcification
            # if filter_calcium:
            calcified_candidates = exam_calcification[exam_calcification >= 130]
            if calcified_candidates.shape[0] == 0:
                continue
            max_HU = calcified_candidates.max()
            if max_HU > max_cac:
                max_cac = max_HU
            # Area in mm^2
            area = calcified_candidates[calcified_candidates != 0].shape[0] * pixel_spacing[0] * pixel_spacing[1]
            # print(area, density_factor(max_HU), max_HU)
            agaston_score += area * density_factor(max_HU)
    print('Max CAC:', max_cac)
    return agaston_score

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    
    # save_path = 'data/EXAMES/Exames_Separados/test_segs'
    
    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['Pacient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    pacients = os.listdir(root_path)
    results = []
    for pacient in pacients:
        print()
        print(pacient)
        # pacient = '180466'
        #! Fake Gated
        if False:
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
            fg_mask_lab = fg_mask_lab_img.get_fdata()#.transpose(1, 2, 0)
            fg_mask_les = fg_mask_les_img.get_fdata()#.transpose(1, 2, 0)
            
            pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            
            roi_score_fg = calculate_score(fg_exam, fg_mask_lab, pixel_spacing)
            les_score_fg = calculate_score(fg_exam, fg_mask_les, pixel_spacing)
            
            # print('Estimated Faked Gated score:', estimated_score_fg)
            # print('Direct Fake Gated Score:', direct_score_fg)
        else:
            roi_score_fg = 0
            les_score_fg = 0
        
        #! Gated
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
        gated_mask_les = gated_mask_les_img.get_fdata()#.transpose(1, 2, 0)
        gated_mask_lab = gated_mask_lab_img.get_fdata()#.transpose(1, 2, 0)
        
        pixel_spacing = gated_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
        
        roi_score_gated = calculate_score(gated_exam, gated_mask_lab, pixel_spacing)
        les_score_gated = calculate_score(gated_exam, gated_mask_les, pixel_spacing)
        
        les_cands_mask = gated_exam.copy()
        les_cands_mask[(les_cands_mask < 130) & (les_cands_mask > 2000)] = 0
        les_cands_mask[(les_cands_mask >= 130) & (les_cands_mask <= 2000)] = 1
        full_score_gated = calculate_score(gated_exam, les_cands_mask, pixel_spacing)
        
        circle_mask = np.zeros(gated_exam.shape[:2])
        circle_mask = cv2.circle(circle_mask, (circle_mask.shape[1] // 2, circle_mask.shape[0] // 2), 160, 1, -1)
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], gated_exam.shape[2], axis=2)
        # Save as nifti
        # new_nifti = nib.Nifti1Image(circle_mask, gated_exam_img.affine)
        # nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/circle_mask_gated.nii.gz')
        circle_cands_exam = gated_exam.copy()
        circle_cands_exam = circle_cands_exam * circle_mask
        circle_cands_exam[(circle_cands_exam) < 130 & (circle_cands_exam > 2000)] = 0
        circle_cands_exam[(circle_cands_exam) >= 130 & (circle_cands_exam <= 2000)] = 1
        circle_mask = circle_cands_exam
        circle_score_gated = calculate_score(gated_exam, circle_mask, pixel_spacing)
        
        print('ROI Score gated:', roi_score_gated)
        print('Lesion Score gated:', les_score_gated)
        print('Full Score Gated:', full_score_gated)
        print('Circle Score Gated:', circle_score_gated)
        print(f'Reference Score: {df_score_ref[df_score_ref["Pacient"] == int(pacient)]["Escore"].values[0]}')

        results.append([pacient, roi_score_gated, roi_score_fg, les_score_gated, les_score_fg, full_score_gated, circle_score_gated])
        # results.append([pacient, estimated_score_gated, estimated_score_fg])

    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    # print(df_score_ref.head())
    df_score_ref['Pacient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    df = pd.DataFrame(results, columns=['Pacient', 'ROI Gated', 'ROI Fake Gated', 'Lesion Gated', 'Lesion Fake Gated', 'Full Mask Gated', 'Circle Mask Gated'])
    # df = pd.DataFrame(results, columns=['Pacient', 'Estimated Score Gated', 'Estimated Score Fake Gated'])
    df['Pacient'] = df['Pacient'].astype(int)
    #df[['ROI Score Gated', 'ROI Score Fake Gated', 'Lesion Score Gated', 'Lesion Score Fake Gated']] = df[['ROI Score Gated', 'ROI Score Fake Gated', 'Lesion Score Gated', 'Lesion Score Fake Gated']].astype(int)
    # Get all the columns that is not in a list
    # exclude_columns = ['Pacient']
    # filtered_columns = [col for col in df.columns if col not in exclude_columns]
    # df[filtered_columns] = df[filtered_columns].astype(int)
    
    df = df.astype(int)
    
    # df[['Estimated Score Gated', 'Estimated Score Fake Gated']] = df[['Estimated Score Gated', 'Estimated Score Fake Gated']].astype(int)
    df = pd.merge(df, df_score_ref[['Pacient', 'Escore']], on='Pacient', how='left')
    
    df = df[['Pacient', 'Escore', 'ROI Gated', 'Lesion Gated', 'Full Mask Gated', 'Circle Mask Gated']]
    
    df['ROI Error'] = df['Escore'] - df['ROI Gated']
    df['Lesion Error'] = df['Escore'] - df['Lesion Gated']
    df['Full Mask Error'] = df['Escore'] - df['Full Mask Gated']
    df['Circle Mask Error'] = df['Escore'] - df['Circle Mask Gated']
    
    average_roi_error = df['ROI Error'].abs().mean()
    average_lesion_error = df['Lesion Error'].abs().mean()
    average_full_mask_error = df['Full Mask Error'].abs().mean()
    average_circle_mask_error = df['Circle Mask Error'].abs().mean()
    
    print('-'*50)
    print('Average ROI Error:', average_roi_error)
    print('Average Lesion Error:', average_lesion_error)
    print('Average Full Mask Error:', average_full_mask_error)
    print('Average Circle Mask Error:', average_circle_mask_error)
    
    df.to_csv('data/EXAMES/calcium_score_estimations.csv', index=False)

    
    
    
    
    
    