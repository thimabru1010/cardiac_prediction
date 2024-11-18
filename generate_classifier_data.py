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

if __name__=='__main__':
    argparser = argparse.ArgumentParser(description='Calculate Agaston Score')
    argparser.add_argument('--root_path', type=str, default='data/EXAMES/Exames_NIFTI', help='Root path of the exams')
    argparser.add_argument('--circle_radius', type=int, default=120, help='Radius for the Circle Mask')
    argparser.add_argument('--fake_gated', action='store_true', help='Use the fake gated exams')
    # argparser.add_argument('--save_path', type=str, help='Path to save the results')
    args = argparser.parse_args()
    
    root_path = args.root_path
    
    df_score_ref = pd.read_excel('data/EXAMES/cac_score_data.xlsx')
    df_score_ref['patient'] = df_score_ref['ID'].astype(int)
    df_score_ref['Escore'] = df_score_ref['Escore'].astype(float)
    
    patients = os.listdir(root_path)
    results = []
    train_circle_data = []
    
    for patient in patients:
        # print(patient)
        if not args.fake_gated:
            exclude_files = ['multi_label', 'multi_lesion', 'binary_lesion']
            keywords_cardiac = ['cardiac']
            exam_basename = get_basename(os.listdir(f'{root_path}/{patient}/{patient}'), exclude_files, keywords_cardiac)
            exam_path = f'{root_path}/{patient}/{patient}/{exam_basename}'
        else:
            exam_path = f'{root_path}/{patient}/{patient}/partes_moles_FakeGated_mean_slice=3mm.nii.gz'
        
        # print(gated_exam_path)
        exam_img = nib.load(exam_path)#.get_fdata()
        
        # ct_data = gated_exam_img.get_fdata()
        exam = exam_img.get_fdata()
        
        circle_mask = np.zeros(exam.shape[:2])
        # circle_mask = np.ones(gated_exam.shape[:2])
        circle_mask = cv2.circle(circle_mask, (circle_mask.shape[1] // 2, circle_mask.shape[0] // 2), args.circle_radius, 1, -1)
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], exam.shape[2], axis=2)
        
        pixel_spacing = exam_img.header.get_zooms()[:3]
        circle_score_gated, conected_lesions, clssf_data = calculate_score(exam, circle_mask, pixel_spacing, patient)
        train_circle_data.append(clssf_data)
        
    train_circle_data = np.concatenate(train_circle_data, axis=0)
    df_classifier_data = pd.DataFrame(train_circle_data, columns=['patient', 'Max HU', 'Centroid X', 'Centroid Y', 'Area', 'Channel'])
    df_classifier_data['patient'] = df_classifier_data['patient'].astype(int)
    df_classifier_data['Escore'] = pd.merge(df_classifier_data, df_score_ref, on='patient', how='left')['Escore']
    df_classifier_data.to_csv(f'data/EXAMES/classifier_dataset_radius={args.circle_radius}.csv', index=False)