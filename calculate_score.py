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

def get_basename(files):
    # Exclude the multi_label, multi_lesion and binary_lesion files with these names inside the list
    exclusion_names=['multi_label', 'multi_lesion', 'binary_lesion', 'cardiac_IncreasedMask', 'HeartSegs', 'cardiac_CalciumCandidates']
    files = [
    file 
    for file in files 
    if not any(f in file for f in exclusion_names)
    ]
    gated_exam_basename = [file for file in files if 'cardiac' in file]
    # exclusion_files = [file for file in gated_exam_basename if any(f in file for f in exclusion_names)]
    # gated_exam_basename.remove(exclusion_files[0])
    # gated_exam_basename.remove(exclusion_files[1])
    # gated_exam_basename.remove(exclusion_files[2])
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

def calculate_score(exam, mask, pixel_spacing, pacient_id=None):
    agaston_score = 0
    max_cac = 0
    # print(mask.shape, exam.shape)
    calc_candidates_mask = exam.copy()
    calc_candidates_mask[calc_candidates_mask < 130] = 0
    calc_candidates_mask[calc_candidates_mask >= 130] = 1
    
    mask[mask != 0] = 1
    calc_candidates_mask = calc_candidates_mask * mask
    
    conected_lesions = np.zeros(mask.shape)
    classification_data = []
    for channel in range(mask.shape[2]):
        _, lesions = cv2.connectedComponents(calc_candidates_mask[:, :, channel].astype(np.uint8))
        conected_lesions[:, :, channel] = lesions
        lesion_labels = list(np.unique(lesions))
        # print(lesions.shape, lesion_labels)
        
        if 0 in lesion_labels:
            lesion_labels.remove(0)
        
        for lesion_label in lesion_labels: # Exclude 0 (background)
            # num_lesion = 2
            # print(lesion_label)
            lesion = lesions.copy()
            lesion[lesion != lesion_label] = 0
            lesion[lesion == lesion_label] = 1
            coordinates = np.argwhere(lesion)
            centroid = np.mean(coordinates, axis=0)
            
            exam_calcification = exam[:, :, channel] * lesion
            calcified_candidates = exam_calcification.copy()    
            if calcified_candidates.shape[0] == 0:
                continue
            max_HU = calcified_candidates.max()
            if max_HU > max_cac:
                max_cac = max_HU
            # Area in mm^2
            area = calcified_candidates[calcified_candidates != 0].shape[0] * pixel_spacing[0] * pixel_spacing[1]
            # print(area)
            # print(area, density_factor(max_HU), max_HU)
            agaston_score += area * density_factor(max_HU)
            
            classification_data.append([pacient_id, max_HU, centroid[0], centroid[1], area, channel])
    # print('Max CAC:', max_cac)
    # print(len(classification_data))
    # 1/0
    return agaston_score, conected_lesions, np.array(classification_data)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Calculate Agaston Score')
    argparser.add_argument('--root_path', type=str, default='data/EXAMES/Exames_NIFTI', help='Root path of the exams')
    argparser.add_argument('--dilate', action='store_true', help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_kernel', type=int, default=7, help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_it', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--circle_radius', type=int, default=120, help='Radius for the Circle Mask')
    # argparser.add_argument('--save_path', type=str, help='Path to save the results')
    args = argparser.parse_args()
    
    root_path = args.root_path
    
    # print(args.dilate)
    # save_path = 'data/EXAMES/Exames_Separados/test_segs'
    
    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['Pacient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    pacients = os.listdir(root_path)
    results = []
    # dilate_kernel = int(args.dilate_kernel)
    kernel = np.ones((args.dilate_kernel, args.dilate_kernel), np.uint8)
    train_circle_data = []
    # circle_radius = 120
    for pacient in pacients:
        # pacient = '176253'
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
            
            print(fg_exam.shape)
            
            pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            
            roi_score_fg, _, _ = calculate_score(fg_exam, fg_mask_lab, pixel_spacing)
            les_score_fg, _, _ = calculate_score(fg_exam, fg_mask_les, pixel_spacing)
            
            # print('Estimated Faked Gated score:', estimated_score_fg)
            # print('Direct Fake Gated Score:', direct_score_fg)
        else:
            roi_score_fg = 0
            les_score_fg = 0
        
        #! Gated
        # print('Gated Agaston Score Calculation')
        gated_exam_basename = get_basename(os.listdir(f'{root_path}/{pacient}/{pacient}'))
        print(gated_exam_basename)
        # print(gated_exam_basename)
        gated_exam_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename}'
        gated_mask_les_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename[:-3]}_multi_lesion.nii.gz'
        gated_mask_lab_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename[:-3]}_multi_label.nii.gz'
        # gated_heart_mask_path = f'{root_path}/{pacient}/{pacient}/cardiac_IncreasedMask.nii.gz'
        
        # print(gated_exam_path)
        gated_exam_img = nib.load(gated_exam_path)#.get_fdata()
        gated_mask_les_img = nib.load(gated_mask_les_path)#.get_fdata()
        gated_mask_lab_img = nib.load(gated_mask_lab_path)#.get_fdata()
        # gated_heart_mask_img = nib.load(gated_heart_mask_path)
        
        print(gated_exam_img.shape)
        
        gated_exam = gated_exam_img.get_fdata()
        gated_mask_les = gated_mask_les_img.get_fdata()#.transpose(1, 2, 0)
        gated_mask_lab = gated_mask_lab_img.get_fdata()#.transpose(1, 2, 0)
        # gated_heart_mask = gated_heart_mask_img.get_fdata()
        
        calc_candidates_mask = gated_exam.copy()
        calc_candidates_mask[calc_candidates_mask < 130] = 0
        calc_candidates_mask[calc_candidates_mask >= 130] = 1
        
        # Create and save a new NIfTI image from the modified data
        new_nifti = nib.Nifti1Image(calc_candidates_mask, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/cardiac_CalciumCandidates_Mask.nii.gz')
        
        pixel_spacing = gated_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
        
        # gated_mask_lab = cv2.dilate(gated_mask_lab, np.ones((5,5), np.uint8), iterations=3)
        if args.dilate:
            gated_mask_lab = cv2.dilate(gated_mask_lab, kernel, iterations=args.dilate_it)
            gated_mask_les = cv2.dilate(gated_mask_les, kernel, iterations=args.dilate_it)
        
        new_nifti = nib.Nifti1Image(gated_mask_lab, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/IncreasedROI_mask_gated.nii.gz')
        new_nifti = nib.Nifti1Image(gated_mask_les, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/IncreasedLesion_mask_gated.nii.gz')
        
        # Calculate the Area of Increasead Lesions Mask
        print(np.unique(gated_mask_les))
        gated_mask_les_tmp = gated_mask_les.copy()
        gated_mask_les_tmp[gated_mask_les_tmp != 0] = 1
        gated_les_area_sum = gated_mask_les_tmp.sum()
        print('Gated Lesion Area:', gated_les_area_sum)
        
        roi_score_gated, connected_lab, _ = calculate_score(gated_exam, gated_mask_lab, pixel_spacing)
        les_score_gated, connected_les, _ = calculate_score(gated_exam, gated_mask_les, pixel_spacing)
        # heart_score_gated = calculate_score(gated_exam, gated_heart_mask, pixel_spacing)
        
        new_nifti = nib.Nifti1Image(connected_lab, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/ROISingleLesions_mask_gated.nii.gz')
        
        new_nifti = nib.Nifti1Image(connected_les, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/LesionSingleLesions_mask_gated.nii.gz')
        
        # les_cands_mask = gated_exam.copy()
        # les_cands_mask[(les_cands_mask < 130) & (les_cands_mask > 2000)] = 0
        # les_cands_mask[(les_cands_mask >= 130) & (les_cands_mask <= 2000)] = 1
        # full_score_gated, _ = calculate_score(gated_exam, les_cands_mask, pixel_spacing)
        
        circle_mask = np.zeros(gated_exam.shape[:2])
        circle_mask = cv2.circle(circle_mask, (circle_mask.shape[1] // 2, circle_mask.shape[0] // 2), args.circle_radius, 1, -1)
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], gated_exam.shape[2], axis=2)
        
        # Save as nifti
        new_nifti = nib.Nifti1Image(circle_mask, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/circle_mask_gated.nii.gz')
        
        new_nifti = nib.Nifti1Image(circle_mask*calc_candidates_mask, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/circle__candidates_mask_gated.nii.gz')
        
        circle_score_gated, conected_lesions, clssf_data = calculate_score(gated_exam, circle_mask, pixel_spacing, pacient)
        new_nifti = nib.Nifti1Image(conected_lesions, gated_exam_img.affine)
        nib.save(new_nifti, f'{root_path}/{pacient}/{pacient}/CircleSingleLesions_mask_gated.nii.gz')
        train_circle_data.append(clssf_data)
        
        
        print('ROI Score gated:', roi_score_gated)
        print('Lesion Score gated:', les_score_gated)
        # print('Full Score Gated:', full_score_gated)
        print('Circle Score Gated:', circle_score_gated)
        print(f'Reference Score: {df_score_ref[df_score_ref["Pacient"] == int(pacient)]["Escore"].values[0]}')
        # print('Heart Score Gated:', heart_score_gated) 

        heart_score_gated = 0
        results.append([pacient, roi_score_gated, roi_score_fg, les_score_gated, les_score_fg, circle_score_gated, gated_les_area_sum])
        # print(results)
        # results.append([pacient, estimated_score_gated, estimated_score_fg])
        # 1/0
    # 1/0
    # classification_data.append([pacient_id, max_HU, centroid[0], centroid[1], area, channel])
    train_circle_data = np.concatenate(train_circle_data, axis=0)
    df_classifier_data = pd.DataFrame(train_circle_data, columns=['Pacient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Channel'])
    df_classifier_data['Pacient'] = df_classifier_data['Pacient'].astype(int)
    # print(train_circle_data.shape)
    # np.save('data/EXAMES/train_circle_data.npy', train_circle_data)
    
    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    # print(df_score_ref.head())
    df_score_ref['Pacient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    df_classifier_data = pd.merge(df_classifier_data, df_score_ref[['Pacient', 'Escore']], on='Pacient', how='left')
    df_classifier_data.to_csv(f'data/EXAMES/classifier_dataset_radius={args.circle_radius}.csv', index=False)
    
    df = pd.DataFrame(results, columns=['Pacient', 'ROI Gated', 'ROI Fake Gated', 'Lesion Gated', 'Lesion Fake Gated', 'Circle Mask Gated', 'Lesion Area Gated'])
    # df = pd.DataFrame(results, columns=['Pacient', 'Estimated Score Gated', 'Estimated Score Fake Gated'])
    print(df.head())
    df['Pacient'] = df['Pacient'].astype(int)
    
    print(df.columns)
    df = df.astype(int)
    
    # df[['Estimated Score Gated', 'Estimated Score Fake Gated']] = df[['Estimated Score Gated', 'Estimated Score Fake Gated']].astype(int)
    df = pd.merge(df, df_score_ref[['Pacient', 'Escore']], on='Pacient', how='left')
    
    df = df[['Pacient', 'Escore', 'ROI Gated', 'Lesion Gated', 'Circle Mask Gated', 'Lesion Area Gated']]
    
    df['ROI Error'] = df['Escore'] - df['ROI Gated']
    df['Lesion Error'] = df['Escore'] - df['Lesion Gated']
    # df['Full Mask Error'] = df['Escore'] - df['Full Mask Gated']
    df['Circle Mask Error'] = df['Escore'] - df['Circle Mask Gated']
    # df['Heart Mask Error'] = df['Escore'] - df['Heart Mask Gated']
    
    # average_roi_error = df['ROI Error'].abs().mean()
    # average_lesion_error = df['Lesion Error'].abs().mean()
    # average_full_mask_error = df['Full Mask Error'].abs().mean()
    # average_circle_mask_error = df['Circle Mask Error'].abs().mean()
    # average_heart_mask_error = df['Heart Mask Error'].abs().mean()
    
    average_roi_error = df['ROI Error'].mean()
    average_lesion_error = df['Lesion Error'].mean()
    # average_full_mask_error = df['Full Mask Error'].mean()
    average_circle_mask_error = df['Circle Mask Error'].mean()
    # average_heart_mask_error = df['Heart Mask Error'].mean()
    
    print('-'*50)
    print('Average ROI Error:', average_roi_error)
    print('Average Lesion Error:', average_lesion_error)
    # print('Average Full Mask Error:', average_full_mask_error)
    print('Average Circle Mask Error:', average_circle_mask_error)
    # print('Average Heart Mask Error:', average_heart_mask_error)
    
    if args.dilate:
        df.to_csv(f'data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations_dilate_it={args.dilate_it}_dilate_k={args.dilate_kernel}.csv', index=False)
    else:
        df.to_csv('data/EXAMES/Calcium_Scores_Estimations/calcium_score_estimations.csv', index=False)

    
    
    
    
    
    