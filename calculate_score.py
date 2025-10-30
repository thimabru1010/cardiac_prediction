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
from utils import get_basename, calculate_area, create_save_nifti
from scipy.ndimage import binary_fill_holes

def calculate_candidates_mask(exam, th=130):
    calc_candidates_mask = exam.copy()
    calc_candidates_mask[calc_candidates_mask < th] = 0
    calc_candidates_mask[calc_candidates_mask >= th] = 1
    return calc_candidates_mask

def save_classifier_data(train_data, df_score_ref, save_path, method='lesion', dilate=None, scores0=False):
    folder_path = f'data/EXAMES/Classifiers_Dataset/{save_path}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # train_data = np.stack(train_data, axis=0)
    train_data = np.concatenate(train_data, axis=0)
    df_classifier_data = pd.DataFrame(train_data, columns=['patient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Area Pixel Spacing', 'Channel'])
    # print(df_classifier_data['patient'])
    df_classifier_data['patient'] = df_classifier_data['patient'].astype(int)
    
    df_classifier_data = pd.merge(df_classifier_data, df_score_ref[['patient', 'Escore']], on='patient', how='left')
    if method == 'lesion':
        if scores0:
            print(os.path.join(folder_path, f'classifier_dataset_lesion_{dilate}.csv'))
            df_classifier_data.to_csv(os.path.join(folder_path, f'classifier_dataset_lesion_{dilate}.csv'), index=False)
        else:
            print(os.path.join(folder_path, f'classifier_dataset_lesion_scores0.csv'))
            df_classifier_data.to_csv(os.path.join(folder_path, f'classifier_dataset_lesion_scores0.csv'), index=False)

def calc_avg_error(df):
    df['Lesion Error'] = df['Escore'] - df['Lesion']
    # df['Heart Mask Error'] = df['Escore'] - df['Heart Mask']
    average_lesion_error = df['Lesion Error'].mean()
    # average_circle_mask_error = df['Heart Mask Error'].mean()
    print('-'*50)
    print('Average Lesion Error:', average_lesion_error)
    # print('Average Heart Mask Error:', average_circle_mask_error)
    return df

def create_circle_mask(center, radius, shape):
    circle_mask = np.zeros(shape[:2], dtype=np.uint8)
    # print(circle_mask.shape)
    cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
    circle_mask = np.repeat(circle_mask[:, :, np.newaxis], shape[2], axis=2)
    return circle_mask

def density_factor(max_HU, min_HU=130):
    if max_HU >= min_HU and max_HU < 200:
        return 1
    elif max_HU >= 200 and max_HU < 300:
        return 2
    elif max_HU >= 300 and max_HU < 400:
        return 3
    elif max_HU >= 400:
        return 4

def calculate_score(exam, mask, bones_mask=None, calc_candidates_mask=None, pixel_spacing=None,\
    patient_id=None, th=130):
    agaston_score = 0
    max_cac = 0
    
    mask[mask != 0] = 1
    
    if bones_mask is None:
        print("No bones mask provided, assuming no bones to exclude.")
        bones_mask = np.zeros(mask.shape)
        
    # Flip the mask to filter the bones and remove them from the mask
    bones_mask = 1 - bones_mask
    # print(bones_mask)
    calc_candidates_mask = calc_candidates_mask * mask  * bones_mask
    
    
    conected_lesions = np.zeros(mask.shape)
    classification_data = []
    for channel in range(mask.shape[2]):
        _, lesions = cv2.connectedComponents(calc_candidates_mask[:, :, channel].astype(np.uint8))
        conected_lesions[:, :, channel] = lesions
        lesion_labels = list(np.unique(lesions))
        
        if 0 in lesion_labels:
            lesion_labels.remove(0)
        
        for lesion_label in lesion_labels: # Exclude 0 (background)
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
            area_pixels = calcified_candidates[calcified_candidates != 0].shape[0]
            area_pixel_spacing = pixel_spacing[0] * pixel_spacing[1]
            area = area_pixels * area_pixel_spacing
            agaston_score += area * density_factor(max_HU, th)
            # print(f"Slice {channel} Score: {area * density_factor(max_HU)}")
            
            # print([patient_id, max_HU, centroid[0], centroid[1], area, channel])
            classification_data.append([patient_id, max_HU, centroid[0], centroid[1], area_pixels, area_pixel_spacing, channel])
        
        if len(lesion_labels) == 0:
            classification_data.append([patient_id, 0, 0, 0, 0, 0, channel])
    return agaston_score, conected_lesions, np.array(classification_data)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Calculate Agaston Score')
    argparser.add_argument('--root_path', type=str, default='data/ExamesArya_NIFTI2', help='Root path of the exams')
    argparser.add_argument('--csv_path', type=str, default='data/cac_score_data.csv', help='Path to the CSV file with reference scores')
    argparser.add_argument('--dilate', action='store_true', help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_kernel', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_it', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--fake_gated', action='store_true', help='Whether to infer fake gated exams')
    argparser.add_argument('--avg', type=int, default=4, help='Number of slices to average')
    argparser.add_argument('--cac_th', type=int, default=130, help='Calcification Threshold in HU')
    argparser.add_argument('--scores0', action='store_true', help='Use only if you are processing scores 0 exams')
    args = argparser.parse_args()
    
    root_path = args.root_path
    
    # df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref = pd.read_csv(args.csv_path)
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore Total'].astype(float)
    
    patients = os.listdir(root_path)
    results = []
    # dilate_kernel = int(args.dilate_kernel)
    kernel = np.ones((args.dilate_kernel, args.dilate_kernel), np.uint8)
    dilate_it = 0
    dilate_k = 0
    if args.dilate:
        dilate_it = args.dilate_it
        dilate_k = args.dilate_kernel
    train_circle_data = []
    train_roi_data = []
    train_les_data = []
    #! Fake Gated
    print('Fake Gated Agaston Score Calculation')
    cont = 0
    avg_str = ''
    avg_flag = False if args.avg == 0 else True
    # exclusion_patients = ['179238', '176064', '177222']
    if args.fake_gated:
        exam_folder = 'Fake_Gated'
        cac_th = args.cac_th
        if avg_flag:
            if args.avg == 4:
                partes_moles_basename = 'non_gated_FakeGated_avg_slices=4'
                partes_moles_heart_filename = 'non_gated_HeartSegs_FakeGated_avg_slices=4.nii.gz'
                partes_moles_bones_filename = 'non_gated_BonesSegs_FakeGated_avg_slices=4.nii.gz'
                avg_str = 'avg=4'
        else:
            partes_moles_basename = 'non_gated_FakeGated'
            partes_moles_heart_filename = 'non_gated_HeartSegs_FakeGated.nii.gz'
            partes_moles_bones_filename = 'non_gated_BonesSegs_FakeGated.nii.gz'
            avg_str = 'All Slices'
        
        for patient in tqdm(patients):
            print(patient)
            
            print(partes_moles_basename)
            fg_exam_path = f'{root_path}/{patient}/{partes_moles_basename}.nii.gz'
            fg_mask_les_path = f'{root_path}/{patient}/{partes_moles_basename}_multi_lesion.nii.gz'
            roi_coronaries_path = f'{root_path}/{patient}/{partes_moles_basename}_multi_label.nii.gz'
            fg_heart_mask_path = f'{root_path}/{patient}/{partes_moles_heart_filename}'
            fg_bones_mask_path = f'{root_path}/{patient}/{partes_moles_bones_filename}'

            fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
            
            print(f'Pixel spacing: {fg_exam_img.header.get_zooms()}')
            
            fg_mask_les_img = nib.load(fg_mask_les_path)#.get_fdata()
            fg_heart_mask_img = nib.load(fg_heart_mask_path)
            fg_bones_mask_img = nib.load(fg_bones_mask_path)
            roi_coronaries_img = nib.load(roi_coronaries_path)
            
            fg_exam = fg_exam_img.get_fdata()
            fg_mask_les = fg_mask_les_img.get_fdata()#.transpose(1, 2, 0)
            fg_heart_mask = fg_heart_mask_img.get_fdata()
            fg_bones_mask = fg_bones_mask_img.get_fdata()
            roi_coronaries_mask = roi_coronaries_img.get_fdata()
            
            roi_coronaries_mask[roi_coronaries_mask != 0] = 1
            
            fg_heart_mask = np.ones_like(fg_exam)
            
            calc_candidates_mask = calculate_candidates_mask(fg_exam, th=cac_th)
            
            create_save_nifti(calc_candidates_mask, fg_exam_img.affine, f'{root_path}/{patient}/{partes_moles_basename}_CalciumCandidates.nii.gz')
            
            if args.dilate:
                fg_mask_les = cv2.dilate(fg_mask_les, kernel, iterations=args.dilate_it)
            
            print(fg_mask_les.shape, fg_heart_mask.shape, fg_bones_mask.shape, calc_candidates_mask.shape)
            create_save_nifti(fg_mask_les, fg_exam_img.affine, f'{root_path}/{patient}/{partes_moles_basename}_multi_lesion_dilate_it={dilate_it}_dilate_k={args.dilate_kernel}.nii.gz')
            create_save_nifti(fg_mask_les*fg_heart_mask*(1 - fg_bones_mask)*calc_candidates_mask, fg_exam_img.affine, f'{root_path}/{patient}/{partes_moles_basename}_lesions_dilate_it={dilate_it}_dilate_k={args.dilate_kernel}_final_mask.nii.gz')
            create_save_nifti(1 - fg_bones_mask, fg_exam_img.affine, f'{root_path}/{patient}/{partes_moles_basename}_NOT_BonesSegs_FakeGated_avg_slices=4.nii.gz')

            pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            les_score_fg, connected_les, les_clssf_data = calculate_score(fg_exam, fg_mask_les, fg_heart_mask, fg_bones_mask, calc_candidates_mask,\
                pixel_spacing, patient_id=patient, th=cac_th)
            
            # roi_coronaries_score_fg, _, _ = calculate_score(fg_exam, roi_coronaries_mask, fg_heart_mask, fg_bones_mask, calc_candidates_mask,\
            #     pixel_spacing, patient_id=patient, th=cac_th)
            
            # heart_score_fg, _, _ = calculate_score(fg_exam, fg_heart_mask, fg_heart_mask, fg_bones_mask, calc_candidates_mask,\
            #     pixel_spacing, patient_id=patient, th=cac_th)
            
            if les_clssf_data.shape[0] != 0:
                train_les_data.append(les_clssf_data)
            
            # print('ROI Score FG:', roi_score_fg)
            print('Lesion Score FG:', les_score_fg)
            # print(f'ROI Coronaries Score FG: {roi_coronaries_score_fg}')
            print(f'Reference Score: {df_score_ref[df_score_ref["patient"] == int(patient)]["Escore"].values[0]}')
            
            # results.append([patient, les_score_fg, roi_coronaries_score_fg, heart_score_fg])
            results.append([patient, les_score_fg])
            
    #! Gated
    if not args.fake_gated:
        exam_folder = 'Gated'
        cac_th = args.cac_th
        print('Gated Agaston Score Calculation')
        exclude_files = ['multi_label', 'multi_lesion', 'binary_lesion', '_CalciumCandidates',\
            '_CircleSingleLesions', '_ROISingleLesions', '_circle_lesions', '_IncreasedLesion',\
                '_clustered=', '_LesionSingleLesions', 'SingleLesions', '_circle', 'non_gated', '_mask']
        keywords_cardiac = ['gated']
        for patient in tqdm(patients):
            print(patient)
            gated_exam_basename = get_basename(os.listdir(f'{root_path}/{patient}'), exclude_files, keywords_cardiac)
            gated_exam_basename = gated_exam_basename.split('.nii.gz')[0]
            print(gated_exam_basename)
            gated_exam_path = f'{root_path}/{patient}/{gated_exam_basename}.nii.gz'
            gated_mask_les_path = f'{root_path}/{patient}/{gated_exam_basename}_multi_lesion.nii.gz'
            # gated_heart_mask_path = f'{root_path}/{patient}/{patient}/cardiac_IncreasedMask.nii.gz'
            
            gated_exam_img = nib.load(gated_exam_path)
            gated_mask_les_img = nib.load(gated_mask_les_path)
            
            gated_exam = gated_exam_img.get_fdata()
            gated_mask_les = gated_mask_les_img.get_fdata()
            
            calc_candidates_mask = calculate_candidates_mask(gated_exam, th=cac_th)
            
            # Create and save a new NIfTI image from the modified data
            create_save_nifti(calc_candidates_mask, gated_exam_img.affine, f'{root_path}/{patient}/gated_CalciumCandidates.nii.gz')

            if args.dilate:
                gated_mask_les = cv2.dilate(gated_mask_les, kernel, iterations=args.dilate_it)

            create_save_nifti(gated_mask_les, gated_exam_img.affine, f'{root_path}/{patient}/gated_IncreasedLesion_mask.nii.gz')
            
            print(gated_mask_les.shape)
            _ = calculate_area(gated_mask_les)
            # print('Gated Lesion Area:', gated_les_area_sum)
            
            pixel_spacing = gated_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            les_score_gated, connected_les, les_clssf_data = calculate_score(
                exam=gated_exam,
                mask=gated_mask_les,
                calc_candidates_mask=calc_candidates_mask,
                pixel_spacing=pixel_spacing,
                patient_id=patient)

            create_save_nifti(connected_les, gated_exam_img.affine, f'{root_path}/{patient}/gated_SingleLesions_mask.nii.gz')

            if les_clssf_data.shape[0] != 0:
                train_les_data.append(les_clssf_data)
            
            print('Lesion Score gated:', les_score_gated)
            print(f'Reference Score: {df_score_ref[df_score_ref["patient"] == int(patient)]["Escore"].values[0]}')

            # print("[DEBUG SHAPES]")
            # print(gated_mask_les.shape, gated_les_area_sum.shape)
            # print(patient, les_score_gated, gated_les_area_sum)
            results.append([patient, les_score_gated])


    # df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref = pd.read_csv(args.csv_path)
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore Total'].astype(float)
    
    if args.dilate:
        save_classifier_data(train_les_data, df_score_ref, save_path=os.path.join(exam_folder, avg_str, str(cac_th)), method='lesion', dilate=f"dilate_it={args.dilate_it}_dilate_k={args.dilate_kernel}", scores0=args.scores0)
    else:
        save_classifier_data(train_les_data, df_score_ref, save_path=os.path.join(exam_folder, avg_str, str(cac_th)), method='lesion', dilate="dilate_it=0_dilate_k=0", scores0=args.scores0)

    print(results)
    results = np.array(results, dtype=int)
    df = pd.DataFrame(results, columns=['patient', 'Lesion'])
    df = pd.merge(df, df_score_ref[['patient', 'Escore']], on='patient', how='left')
    df = df[['patient', 'Escore', 'Lesion']]
    
    df = calc_avg_error(df)
    
    cac_estimations_path = f'data/Calcium_Scores_Estimations/{exam_folder}/{avg_str}/{cac_th}'
    if not os.path.exists(cac_estimations_path):
        os.makedirs(cac_estimations_path)

    if args.scores0:
        df.to_csv(os.path.join(cac_estimations_path, f'calcium_score_estimations_scores0.csv'), index=False)
    else:
        if args.dilate:
            df.to_csv(os.path.join(cac_estimations_path, f'calcium_score_estimations_dilate_it={args.dilate_it}_dilate_k={args.dilate_kernel}_{avg_str}.csv'), index=False)
        else:
            df.to_csv(os.path.join(cac_estimations_path, f'calcium_score_estimations_{avg_str}.csv'), index=False)

    
    
    
    
    
    