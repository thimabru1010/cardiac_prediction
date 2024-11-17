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

def calculate_candidates_mask(exam):
    calc_candidates_mask = exam.copy()
    calc_candidates_mask[calc_candidates_mask < 130] = 0
    calc_candidates_mask[calc_candidates_mask >= 130] = 1
    return calc_candidates_mask
            
def save_classifier_data(train_circle_data, df_score_ref, exam_folder):
    train_circle_data = np.concatenate(train_circle_data, axis=0)
    df_classifier_data = pd.DataFrame(train_circle_data, columns=['patient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Channel'])
    df_classifier_data['patient'] = df_classifier_data['patient'].astype(int)
    
    df_classifier_data = pd.merge(df_classifier_data, df_score_ref[['patient', 'Escore']], on='patient', how='left')
    df_classifier_data.to_csv(f'data/EXAMES/Classifiers_Dataset/{exam_folder}/classifier_dataset_radius={args.circle_radius}.csv', index=False)
        
def print_avg_error(df):
    df['ROI Error'] = df['Escore'] - df['ROI']
    df['Lesion Error'] = df['Escore'] - df['Lesion']
    df['Circle Mask Error'] = df['Escore'] - df['Circle Mask']
    average_roi_error = df['ROI Error'].mean()
    average_lesion_error = df['Lesion Error'].mean()
    average_circle_mask_error = df['Circle Mask Error'].mean()
    print('-'*50)
    print('Average ROI Error:', average_roi_error)
    print('Average Lesion Error:', average_lesion_error)
    print('Average Circle Mask Error:', average_circle_mask_error)

def create_circle_mask(center, radius, shape):
    circle_mask = np.zeros(shape[:2], dtype=np.uint8)
    # print(circle_mask.shape)
    cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
    circle_mask = np.repeat(circle_mask[:, :, np.newaxis], shape[2], axis=2)
    return circle_mask
        
def calculate_area(mask):
    mask_tmp = mask.copy()
    mask_tmp[mask_tmp != 0] = 1
    return mask_tmp.sum()
        
def create_save_nifti(data, affine, output_path):
    new_nifti = nib.Nifti1Image(data, affine)
    nib.save(new_nifti, output_path)
    
def get_basename(files, exclude_files, keywords):
    # Exclude files based on the exclude_files list
    # files = [file for file in files if not any(f in file for f in exclude_files)]
    files = [file for file in files if all(f in file for f in exclude_files)]

    if gated_exam_basename := [
        file for file in files if any(keyword in file for keyword in keywords)]:
        return gated_exam_basename[0]
    else:
        raise ValueError("No matching files found.")

def density_factor(max_HU):
    if max_HU >= 130 and max_HU < 200:
        return 1
    elif max_HU >= 200 and max_HU < 300:
        return 2
    elif max_HU >= 300 and max_HU < 400:
        return 3
    elif max_HU >= 400:
        return 4

def calculate_score(exam, mask, pixel_spacing, patient_id=None):
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
            
            classification_data.append([patient_id, max_HU, centroid[0], centroid[1], area, channel])
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
    argparser.add_argument('--partes_moles', action='store_true', help='Whether to infer partes_moles exams')
    args = argparser.parse_args()
    
    root_path = args.root_path
    
    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    patients = os.listdir(root_path)
    results = []
    # dilate_kernel = int(args.dilate_kernel)
    kernel = np.ones((args.dilate_kernel, args.dilate_kernel), np.uint8)
    train_circle_data = []
    #! Fake Gated
    print('Fake Gated Agaston Score Calculation')
    if args.partes_moles:
        exam_folder = 'Fake_Gated'
        exclude_files = ['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
        keywords_partes_moles = ['partes_moles_body', 'mediastino']
        for patient in patients:
            print(patient)
            fg_exam_path = f'{root_path}/{patient}/{patient}/partes_moles_FakeGated_mean_slice=3mm.nii.gz'
            fg_mask_lab_path = f'{root_path}/{patient}/{patient}/partes_moles_FakeGated_mean_slice=3mm_multi_label.nii.gz'
            fg_mask_les_path = f'{root_path}/{patient}/{patient}/partes_moles_FakeGated_mean_slice=3mm_multi_lesion.nii.gz'
            
            fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
            fg_mask_lab_img = nib.load(fg_mask_lab_path)#.get_fdata()
            fg_mask_les_img = nib.load(fg_mask_les_path)#.get_fdata()
            
            fg_exam = fg_exam_img.get_fdata()
            fg_mask_lab = fg_mask_lab_img.get_fdata()#.transpose(1, 2, 0)
            fg_mask_les = fg_mask_les_img.get_fdata()#.transpose(1, 2, 0)
            
            calc_candidates_mask = calculate_candidates_mask(fg_exam)
            
            create_save_nifti(calc_candidates_mask, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_CalciumCandidates.nii.gz')
            
            if args.dilate:
                fg_mask_les = cv2.dilate(fg_mask_les, kernel, iterations=args.dilate_it)
            
            create_save_nifti(fg_mask_les, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_IncreasedLesion_mask.nii.gz')
            
            fg_les_area_sum = calculate_area(fg_mask_les)
            print('Fake Gated Lesion Area:', fg_les_area_sum)
            
            pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            roi_score_fg, connected_lab, _ = calculate_score(fg_exam, fg_mask_lab, pixel_spacing)
            les_score_fg, connected_les, _ = calculate_score(fg_exam, fg_mask_les, pixel_spacing)
            
            create_save_nifti(connected_lab, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_ROISingleLesions_mask.nii.gz')
            create_save_nifti(connected_les, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_LesionSingleLesions_mask.nii.gz')
            
            circle_mask = create_circle_mask((fg_exam.shape[1] // 2, fg_exam.shape[0] // 2), args.circle_radius, fg_exam.shape)
            
            create_save_nifti(circle_mask, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_circle_mask.nii.gz')
            create_save_nifti(circle_mask*fg_mask_les, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_circle_lesions_mask.nii.gz')
            
            circle_score_fg, conected_circle_lesions, clssf_data = calculate_score(fg_exam, circle_mask, pixel_spacing, patient)
            create_save_nifti(conected_circle_lesions, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_CircleSingleLesions_mask.nii.gz')
            
            train_circle_data.append(clssf_data)
            
            print('ROI Score FG:', roi_score_fg)
            print('Lesion Score FG:', les_score_fg)
            print('Circle Score FG:', circle_score_fg)
            print(f'Reference Score: {df_score_ref[df_score_ref["patient"] == int(patient)]["Escore"].values[0]}')
            
            results.append([patient, roi_score_fg, les_score_fg, circle_score_fg, fg_les_area_sum])
            
            
    #! Gated
    if not args.partes_moles:
        exam_folder = 'Gated'
        print('Gated Agaston Score Calculation')
        exclude_files = ['multi_label', 'multi_lesion', 'binary_lesion']
        keywords_cardiac = ['cardiac']
        for patient in patients:
            print(patient)
            gated_exam_basename = get_basename(os.listdir(f'{root_path}/{patient}/{patient}'))
            print(gated_exam_basename)
            gated_exam_path = f'{root_path}/{patient}/{patient}/{gated_exam_basename}'
            gated_mask_les_path = f'{root_path}/{patient}/{patient}/{gated_exam_basename[:-3]}_multi_lesion.nii.gz'
            gated_mask_lab_path = f'{root_path}/{patient}/{patient}/{gated_exam_basename[:-3]}_multi_label.nii.gz'
            # gated_heart_mask_path = f'{root_path}/{patient}/{patient}/cardiac_IncreasedMask.nii.gz'
            
            # print(gated_exam_path)
            gated_exam_img = nib.load(gated_exam_path)
            gated_mask_les_img = nib.load(gated_mask_les_path)
            gated_mask_lab_img = nib.load(gated_mask_lab_path)
            
            gated_exam = gated_exam_img.get_fdata()
            gated_mask_les = gated_mask_les_img.get_fdata()
            gated_mask_lab = gated_mask_lab_img.get_fdata()
            
            calc_candidates_mask = calculate_candidates_mask(gated_exam)
            
            # Create and save a new NIfTI image from the modified data
            create_save_nifti(calc_candidates_mask, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_CalciumCandidates.nii.gz')

            if args.dilate:
                # gated_mask_lab = cv2.dilate(gated_mask_lab, kernel, iterations=args.dilate_it)
                gated_mask_les = cv2.dilate(gated_mask_les, kernel, iterations=args.dilate_it)
            
            create_save_nifti(gated_mask_les, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_IncreasedLesion_mask.nii.gz')
            
            gated_les_area_sum = calculate_area(gated_mask_les)
            print('Gated Lesion Area:', gated_les_area_sum)
            
            pixel_spacing = gated_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            roi_score_gated, connected_lab, _ = calculate_score(gated_exam, gated_mask_lab, pixel_spacing)
            les_score_gated, connected_les, _ = calculate_score(gated_exam, gated_mask_les, pixel_spacing)
            
            create_save_nifti(connected_lab, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_ROISingleLesions_mask.nii.gz')
            create_save_nifti(connected_les, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_LesionSingleLesions_mask.nii.gz')

            circle_mask = create_circle_mask((gated_exam.shape[1] // 2, gated_exam.shape[0] // 2), args.circle_radius, gated_exam.shape)

            create_save_nifti(circle_mask, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_circle_mask.nii.gz')
            create_save_nifti(circle_mask*gated_mask_les, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_circle_lesions_mask.nii.gz')
            
            circle_score_gated, conected_circle_lesions, clssf_data = calculate_score(gated_exam, circle_mask, pixel_spacing, patient)
            create_save_nifti(conected_circle_lesions, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_CircleSingleLesions_mask.nii.gz')

            train_circle_data.append(clssf_data)
            
            print('ROI Score gated:', roi_score_gated)
            print('Lesion Score gated:', les_score_gated)
            print('Circle Score Gated:', circle_score_gated)
            print(f'Reference Score: {df_score_ref[df_score_ref["patient"] == int(patient)]["Escore"].values[0]}')

            results.append([patient, roi_score_gated, les_score_gated, circle_score_gated, gated_les_area_sum])


    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    save_classifier_data(train_circle_data, df_score_ref, exam_folder)
    
    results = np.array(results, dtype=int)
    df = pd.DataFrame(results, columns=['patient', 'ROI', 'Lesion', 'Circle Mask', 'Lesion Area'])
    df = pd.merge(df, df_score_ref[['patient', 'Escore']], on='patient', how='left')
    df = df[['patient', 'Escore', 'ROI', 'Lesion', 'Circle Mask', 'Lesion Area']]
    
    if args.dilate:
        df.to_csv(f'data/EXAMES/Calcium_Scores_Estimations/{exam_folder}/calcium_score_estimations_dilate_it={args.dilate_it}_dilate_k={args.dilate_kernel}.csv', index=False)
    else:
        df.to_csv(f'data/EXAMES/Calcium_Scores_Estimations/{exam_folder}/calcium_score_estimations.csv', index=False)
        
    print_avg_error(df)

    
    
    
    
    
    