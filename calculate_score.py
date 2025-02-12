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
            
def save_classifier_data(train_data, df_score_ref, exam_folder, method='circle', dilate=None):
    folder_path = f'data/EXAMES/Classifiers_Dataset/{exam_folder}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # train_data = np.stack(train_data, axis=0)
    train_data = np.concatenate(train_data, axis=0)
    df_classifier_data = pd.DataFrame(train_data, columns=['patient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Area Pixel Spacing', 'Channel'])
    # print(df_classifier_data['patient'])
    df_classifier_data['patient'] = df_classifier_data['patient'].astype(int)
    
    df_classifier_data = pd.merge(df_classifier_data, df_score_ref[['patient', 'Escore']], on='patient', how='left')
    if method == 'circle':
        df_classifier_data.to_csv(os.path.join(folder_path, f'classifier_dataset_radius={args.circle_radius}.csv'), index=False)
    elif method == 'roi':
        df_classifier_data.to_csv(os.path.join(folder_path, 'classifier_dataset_roi.csv'), index=False)
    elif method == 'lesion':
        print(os.path.join(folder_path, f'classifier_dataset_lesion_{dilate}.csv'))
        df_classifier_data.to_csv(os.path.join(folder_path, f'classifier_dataset_lesion_{dilate}.csv'), index=False)
        
def calc_avg_error(df):
    df['Lesion Error'] = df['Escore'] - df['Lesion']
    df['Heart Mask Error'] = df['Escore'] - df['Heart Mask']
    average_lesion_error = df['Lesion Error'].mean()
    average_circle_mask_error = df['Heart Mask Error'].mean()
    print('-'*50)
    print('Average Lesion Error:', average_lesion_error)
    print('Average Heart Mask Error:', average_circle_mask_error)
    return df

def create_circle_mask(center, radius, shape):
    circle_mask = np.zeros(shape[:2], dtype=np.uint8)
    # print(circle_mask.shape)
    cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
    circle_mask = np.repeat(circle_mask[:, :, np.newaxis], shape[2], axis=2)
    return circle_mask

def density_factor(max_HU):
    if max_HU >= 100 and max_HU < 200:
        return 1
    elif max_HU >= 200 and max_HU < 300:
        return 2
    elif max_HU >= 300 and max_HU < 400:
        return 3
    elif max_HU >= 400:
        return 4

def calculate_score(exam, mask, heart_roi_mask, bones_mask, calc_candidates_mask, pixel_spacing, patient_id=None, th=130):
    agaston_score = 0
    max_cac = 0
    
    mask[mask != 0] = 1
    
    # Flip the mask to filter the bones and remove them from the mask
    bones_mask = 1 - bones_mask
    
    if heart_roi_mask.shape[0] != mask.shape[0] or heart_roi_mask.shape[1] != mask.shape[1]:
        heart_roi_mask_tmp = np.zeros((exam.shape[0], exam.shape[1], exam.shape[2]))
        for i in range(heart_roi_mask.shape[2]):
            image_tmp = heart_roi_mask[:, :, i]
            image_tmp = cv2.resize(image_tmp, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            heart_roi_mask_tmp[:, :, i] = binary_fill_holes(image_tmp)
        heart_roi_mask = heart_roi_mask_tmp.copy()
        
    print(mask.shape, calc_candidates_mask.shape, heart_roi_mask.shape)
    calc_candidates_mask = calc_candidates_mask * mask * heart_roi_mask * bones_mask
    
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
            agaston_score += area * density_factor(max_HU)
            # print(f"Slice {channel} Score: {area * density_factor(max_HU)}")
            
            # print([patient_id, max_HU, centroid[0], centroid[1], area, channel])
            classification_data.append([patient_id, max_HU, centroid[0], centroid[1], area_pixels, area_pixel_spacing, channel])
        
        if len(lesion_labels) == 0:
            classification_data.append([patient_id, 0, 0, 0, 0, 0, channel])
    return agaston_score, conected_lesions, np.array(classification_data)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Calculate Agaston Score')
    argparser.add_argument('--root_path', type=str, default='data/EXAMES/Exames_NIFTI', help='Root path of the exams')
    argparser.add_argument('--dilate', action='store_true', help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_kernel', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--dilate_it', type=int, default=5, help='Wheter to dilate the mask or not')
    argparser.add_argument('--circle_radius', type=int, default=120, help='Radius for the Circle Mask')
    argparser.add_argument('--partes_moles', action='store_true', help='Whether to infer partes_moles exams')
    # argparser.add_argument('--avg3', action='store_true', help='Whether to use partes_moles exams with 3mm slice thickness')
    argparser.add_argument('--avg', type=int, default=0, help='Number of slices to average')
    argparser.add_argument('--cac_th', type=int, default=130, help='Calcification Threshold in HU')
    argparser.add_argument('--seg_model', type=str, default='lesion', choices=['artery', 'lesion'], help='Segmentation model to use')
    args = argparser.parse_args()
    
    root_path = args.root_path
    
    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
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
    exclusion_patients = ['179238', '176064', '177222']
    if args.partes_moles:
        exam_folder = 'Fake_Gated'
        cac_th = args.cac_th
        exclude_files = ['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
        keywords_partes_moles = ['partes_moles_body', 'mediastino']
        if avg_flag:
            if args.avg == 4:
                partes_moles_basename = 'partes_moles_FakeGated_avg_slices=4'
                partes_moles_heart_filename = 'partes_moles_HeartSegs_FakeGated_avg_slices=4.nii.gz'
                partes_moles_bones_filename = 'partes_moles_BonesSegs_FakeGated_avg_slices=4.nii.gz'
                partes_moles_coronary_arteries_filename = 'partes_moles_FakeGated_avg_slices=4_CoronaryArteries.nii.gz'
                avg_str = 'avg=4'
            elif args.avg == 3:
                partes_moles_basename = 'partes_moles_FakeGated_mean_slice=3mm'
                partes_moles_heart_filename = 'partes_moles_HeartSegs_FakeGated_avg_slices=3.nii.gz'
                avg_str = 'avg=3'
        else:
            partes_moles_basename = 'partes_moles_FakeGated'
            partes_moles_heart_filename = 'partes_moles_HeartSegs_FakeGated.nii.gz'
            partes_moles_bones_filename = 'partes_moles_BonesSegs_FakeGated.nii.gz'
            avg_str = 'All Slices'
        
        for patient in tqdm(patients):
            print(patient)
            
            if patient in exclusion_patients:
                continue
            
            fg_exam_path = f'{root_path}/{patient}/{patient}/{partes_moles_basename}.nii.gz'
            if args.seg_model == 'artery':
                fg_mask_les_path = f'{root_path}/{patient}/{patient}/{partes_moles_coronary_arteries_filename}'
            else:
                fg_mask_les_path = f'{root_path}/{patient}/{patient}/{partes_moles_basename}_multi_lesion.nii.gz'
            # fg_heart_mask_path = f'{root_path}/{patient}/{patient}/{partes_moles_heart_filename}'
            fg_bones_mask_path = f'{root_path}/{patient}/{patient}/{partes_moles_bones_filename}'
            
            fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
            fg_mask_les_img = nib.load(fg_mask_les_path)#.get_fdata()
            # fg_heart_mask_img = nib.load(fg_heart_mask_path)
            fg_bones_mask_img = nib.load(fg_bones_mask_path)
            
            fg_exam = fg_exam_img.get_fdata()
            fg_mask_les = fg_mask_les_img.get_fdata()#.transpose(1, 2, 0)
            # fg_heart_mask = fg_heart_mask_img.get_fdata()
            fg_bones_mask = fg_bones_mask_img.get_fdata()
            
            fg_heart_mask = np.ones_like(fg_exam)
            
            calc_candidates_mask = calculate_candidates_mask(fg_exam, th=cac_th)
            
            create_save_nifti(calc_candidates_mask, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_CalciumCandidates.nii.gz')
            
            if args.dilate:
                fg_mask_les = cv2.dilate(fg_mask_les, kernel, iterations=args.dilate_it)
            
            create_save_nifti(fg_mask_les, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_multi_lesion_dilate_it={dilate_it}_dilate_k={args.dilate_kernel}.nii.gz')
            create_save_nifti(fg_mask_les*fg_heart_mask*(1 - fg_bones_mask)*calc_candidates_mask, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_lesions_dilate_it={dilate_it}_dilate_k={args.dilate_kernel}_final_mask.nii.gz')
            create_save_nifti(1 - fg_bones_mask, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/partes_moles_NOT_BonesSegs_FakeGated_avg_slices=4')
            
            pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            les_score_fg, connected_les, les_clssf_data = calculate_score(fg_exam, fg_mask_les, fg_heart_mask, fg_bones_mask, calc_candidates_mask,\
                pixel_spacing, patient_id=patient, th=cac_th)
            
            # create_save_nifti(connected_les, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_LesionSingleLesions_mask.nii.gz')
            
            # circle_mask = create_circle_mask((fg_exam.shape[1] // 2, fg_exam.shape[0] // 2), args.circle_radius, fg_exam.shape)
            
            # create_save_nifti(fg_heart_mask, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_circle_mask.nii.gz')
            # create_save_nifti(fg_heart_mask*fg_mask_les, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_heart_lesions_mask.nii.gz')
            
            circle_score_fg, conected_heart_lesions, clssf_data = calculate_score(fg_exam, fg_heart_mask, fg_heart_mask, fg_bones_mask, calc_candidates_mask,\
                pixel_spacing, patient)
            # create_save_nifti(conected_heart_lesions, fg_exam_img.affine, f'{root_path}/{patient}/{patient}/{partes_moles_basename}_HeartSingleLesions_mask.nii.gz')
            
            if clssf_data.shape[0] != 0:
                train_circle_data.append(clssf_data)
            if les_clssf_data.shape[0] != 0:
                train_les_data.append(les_clssf_data)
            
            # print('ROI Score FG:', roi_score_fg)
            print('Lesion Score FG:', les_score_fg)
            print('Circle Score FG:', circle_score_fg)
            print(f'Reference Score: {df_score_ref[df_score_ref["patient"] == int(patient)]["Escore"].values[0]}')
            
            results.append([patient, les_score_fg, circle_score_fg])
            
    #! Gated
    if not args.partes_moles:
        exam_folder = 'Gated'
        cac_th = args.cac_th
        print('Gated Agaston Score Calculation')
        exclude_files = ['multi_label', 'multi_lesion', 'binary_lesion', 'cardiac_CalciumCandidates',\
            'cardiac_CircleSingleLesions', 'cardiac_ROISingleLesions', 'cardiac_circle_lesions', 'cardiac_IncreasedLesion',\
                'cardiac_clustered=', 'cardiac_LesionSingleLesions', 'cardiac_circle']
        keywords_cardiac = ['cardiac']
        for patient in tqdm(patients):
            print(patient)
            gated_exam_basename = get_basename(os.listdir(f'{root_path}/{patient}/{patient}'), exclude_files, keywords_cardiac)
            print(gated_exam_basename)
            gated_exam_path = f'{root_path}/{patient}/{patient}/{gated_exam_basename}'
            gated_mask_les_path = f'{root_path}/{patient}/{patient}/{gated_exam_basename[:-3]}_multi_lesion.nii.gz'
            # gated_heart_mask_path = f'{root_path}/{patient}/{patient}/cardiac_IncreasedMask.nii.gz'
            
            gated_exam_img = nib.load(gated_exam_path)
            gated_mask_les_img = nib.load(gated_mask_les_path)
            
            gated_exam = gated_exam_img.get_fdata()
            gated_mask_les = gated_mask_les_img.get_fdata()
            
            calc_candidates_mask = calculate_candidates_mask(gated_exam, th=cac_th)
            
            # Create and save a new NIfTI image from the modified data
            create_save_nifti(calc_candidates_mask, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_CalciumCandidates.nii.gz')

            if args.dilate:
                gated_mask_les = cv2.dilate(gated_mask_les, kernel, iterations=args.dilate_it)
            
            create_save_nifti(gated_mask_les, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_IncreasedLesion_mask.nii.gz')
            
            gated_les_area_sum = calculate_area(gated_mask_les)
            print('Gated Lesion Area:', gated_les_area_sum)
            
            pixel_spacing = gated_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
            les_score_gated, connected_les, les_clssf_data = calculate_score(gated_exam, gated_mask_les, pixel_spacing, patient_id=patient)
            
            create_save_nifti(connected_les, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_LesionSingleLesions_mask.nii.gz')

            circle_mask = create_circle_mask((gated_exam.shape[1] // 2, gated_exam.shape[0] // 2), args.circle_radius, gated_exam.shape)

            # create_save_nifti(circle_mask, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_circle_mask.nii.gz')
            # create_save_nifti(circle_mask*gated_mask_les, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_circle_lesions_mask.nii.gz')
            
            circle_score_gated, conected_circle_lesions, clssf_data = calculate_score(gated_exam, circle_mask, pixel_spacing, patient)
            create_save_nifti(conected_circle_lesions, gated_exam_img.affine, f'{root_path}/{patient}/{patient}/cardiac_CircleSingleLesions_mask.nii.gz')

            if clssf_data.shape[0] != 0:
                train_circle_data.append(clssf_data)
            if les_clssf_data.shape[0] != 0:
                train_les_data.append(les_clssf_data)
            
            print('Lesion Score gated:', les_score_gated)
            print('Circle Score Gated:', circle_score_gated)
            print(f'Reference Score: {df_score_ref[df_score_ref["patient"] == int(patient)]["Escore"].values[0]}')

            results.append([patient, les_score_gated, circle_score_gated, gated_les_area_sum])


    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    save_classifier_data(train_circle_data, df_score_ref, exam_folder, method='circle')
    if args.dilate:
        save_classifier_data(train_les_data, df_score_ref, os.path.join(exam_folder, avg_str, str(cac_th)), method='lesion', dilate=f"dilate_it={args.dilate_it}_dilate_k={args.dilate_kernel}")
    else:
        save_classifier_data(train_les_data, df_score_ref, os.path.join(exam_folder, avg_str, str(cac_th)), method='lesion', dilate=f"dilate_it=0_dilate_k=0")
        
    results = np.array(results, dtype=int)
    df = pd.DataFrame(results, columns=['patient', 'Lesion', 'Heart Mask'])
    df = pd.merge(df, df_score_ref[['patient', 'Escore']], on='patient', how='left')
    df = df[['patient', 'Escore', 'Lesion', 'Heart Mask']]
    
    df = calc_avg_error(df)
    
    cac_estimations_path = f'data/EXAMES/Calcium_Scores_Estimations/{exam_folder}/{avg_str}/{cac_th}/{args.seg_model}'
    if not os.path.exists(cac_estimations_path):
        os.makedirs(cac_estimations_path)
            
    if args.dilate:
        df.to_csv(os.path.join(cac_estimations_path, f'calcium_score_estimations_dilate_it={args.dilate_it}_dilate_k={args.dilate_kernel}_{avg_str}.csv'), index=False)
    else:
        df.to_csv(os.path.join(cac_estimations_path, f'calcium_score_estimations_{avg_str}.csv'), index=False)

    
    
    
    
    
    