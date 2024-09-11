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

if __name__ == '__main__':
    #! Estimated Calculation of the Agaston score
    root_path = 'data/EXAMES/Exames_NIFTI'
    # fg_mask_path = 'data/EXAMES/Exames_Separados/ALL_FakeGated_Preds/cardiac_circle_FakeGated.nii_multi_label.nii.gz'
    # fg_exam_path = 'data/EXAMES/Exames_Separados/ALL_FakeGated/cardiac_circle_FakeGated.nii.gz'
    # Save
    save_path = 'data/EXAMES/Exames_Separados/test_segs'
        
    pacients = os.listdir(root_path)
    for pacient in pacients:
        print(pacient)
        fg_mask_path = f'{root_path}/{pacient}/{pacient}/partes_moles_FakeGated.nii_multi_label.nii.gz'
        fg_exam_path = f'{root_path}/{pacient}/{pacient}/partes_moles_FakeGated.nii.gz'
        
        fg_mask_img = nib.load(fg_mask_path)#.get_fdata()
        fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
        
        fg_mask = fg_mask_img.get_fdata().transpose(1, 2, 0)
        fg_exam = fg_exam_img.get_fdata()
        
        print(fg_mask.shape)
        print(fg_exam.shape)
        
        labels = np.unique(fg_mask)
        print(labels)
        
        fg_mask[fg_mask != 0] = 1
        fg_exam_calcification = fg_exam * fg_mask
        print(fg_exam_calcification.max())
        calcified_candidates = fg_exam_calcification[fg_exam_calcification >= 130]
        # Extract pixel spacing and slice thickness
        pixel_spacing = fg_exam_img.header.get_zooms()[:3]  # (x, y, z) spacing
        slice_thickness = pixel_spacing[2]
        print('Pixel Spacing:', pixel_spacing)
        print(pixel_spacing[0] == pixel_spacing[1])

        print(calcified_candidates.shape)
        score = calcified_candidates.sum() * pixel_spacing[0] * pixel_spacing[1] * slice_thickness
        print('Estimated score:', score)
        
        # new_nifti = nib.Nifti1Image(calcified_candidates, fg_exam_img.affine)
        # nib.save(new_nifti, f'{save_path}/calcification.nii.gz')
        
        #! Direct Calculation of the Agaston score
        # fg_mask_path = 'data/EXAMES/Exames_Separados/ALL_FakeGated_Preds/cardio_circle_mask.nii_multi_lesion.nrrd'
        fg_mask_path = 'data/EXAMES/Exames_Separados/ALL_FakeGated_Preds/cardiac_circle_FakeGated.nii_multi_lesion.nii.gz'
        # fg_exam_path = 'data/EXAMES/Exames_Separados/ALL_FakeGated/cardiac_circle_FakeGated.nii.gz'
        
        # fg_mask = sitk.GetArrayFromImage(sitk.ReadImage(fg_mask_path)).transpose(1, 2, 0)
        fg_mask_img = nib.load(fg_mask_path)#.get_fdata()
        fg_exam_img = nib.load(fg_exam_path)#.get_fdata()
        
        fg_mask = fg_mask_img.get_fdata()
        fg_exam = fg_exam_img.get_fdata()
        print(fg_mask.shape)
        print(fg_exam.shape)
        
        fg_mask[fg_mask != 0] = 1
        fg_exam_calcif = fg_exam * fg_mask
        
        score = fg_exam_calcif.sum() / fg_mask[fg_mask == 1].shape[0]
        
        print('Direct Score:', score)
        
        #! Direct Baseline with gated exam
        gated_exam_path = 'data/EXAMES/Exames_Separados/ALL/61113_0_2_cardiac_30.nii.gz'
        gated_mask_path = 'data/EXAMES/Exames_Separados/ALL_preds/61113_0_2_cardiac_30.nii_multi_lesion.nii.gz'
        
        gated_exam_img = nib.load(gated_exam_path)#.get_fdata()
        # gated_mask = sitk.GetArrayFromImage(sitk.ReadImage(gated_mask_path)).transpose(1, 2, 0)
        gated_mask_img = nib.load(gated_mask_path)#.get_fdata()
        
        gated_exam = gated_exam_img.get_fdata()
        gated_mask = gated_mask_img.get_fdata()
        print(gated_mask.shape)
        labels = np.unique(gated_mask)
        print(labels)
        
        # print(nib.load(gated_exam_path).affine.shape)
        
        # Dilate the mask
        # kernel = np.ones((10,10), np.uint8)
        # gated_mask = cv2.dilate(gated_mask, kernel, iterations=3)
        
        gated_mask[gated_mask != 0] = 1
        gated_exam_calcif = gated_exam * gated_mask
        
        print(gated_exam_calcif.max())
        
        score_gated = gated_exam_calcif.sum() / gated_mask[gated_mask == 1].shape[0]
        print('Direct Score gated:', score_gated)
        
        #! Estimated Baseline with gated exam
        gated_exam_path = 'data/EXAMES/Exames_Separados/ALL/61113_0_2_cardiac_30.nii.gz'
        gated_mask_path = 'data/EXAMES/Exames_Separados/ALL_preds/61113_0_2_cardiac_30.nii_multi_label.nii.gz'
        
        gated_exam_img = nib.load(gated_exam_path)#.get_fdata()
        # gated_mask = sitk.GetArrayFromImage(sitk.ReadImage(gated_mask_path)).transpose(1, 2, 0)
        gated_mask_img = nib.load(gated_mask_path)#.get_fdata()
        
        gated_exam = gated_exam_img.get_fdata()
        gated_mask = gated_mask_img.get_fdata()
        print(gated_mask.shape)
        labels = np.unique(gated_mask)
        print(labels)
        
        # print(nib.load(gated_exam_path).affine.shape)
        
        # Dilate the mask
        # kernel = np.ones((10,10), np.uint8)
        # gated_mask = cv2.dilate(gated_mask, kernel, iterations=3)
        
        gated_mask[gated_mask != 0] = 1
        gated_exam_calcif = gated_exam * gated_mask
        calcified_candidates = gated_exam_calcif[gated_exam_calcif >= 130]
        
        print(gated_exam_calcif.max())
        
        score_gated = calcified_candidates.sum() / calcified_candidates.shape[0]
        print('Estimated Score gated:', score_gated)

    
    
    
    
    
    