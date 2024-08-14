import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti

if __name__ == '__main__':
    #TODO: Aplicar a segmentação em cada um dos niftis
    input_path = 'data/EXAMES/Exames_Separados/ALL/61113_0_5_partes_moles__10.nii.gz'
    # input_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/61113/Nifti/61113_0/2_cardiac_30.nii.gz'
    output_path = 'data/EXAMES/Exames_Separados/ALL_TotalSeg'
    
    # input_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/122932/Nifti/122932_0/3_escore_de_calcio.nii.gz'
    # output_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/122932/Nifti/122932_0'
    
    input_img = nib.load(input_path)#.get_fdata()
    # os.system(f'TotalSegmentator -i {input_path} -o {os.path.join(output_path, "segs3")} -t coronary_arteries')

    output_img = totalsegmentator(input_img, task='total')
    # output_img = nib.load(f'{output_path}/cardio_segs.nii.gz')
    
    print('-'*50)
    output_data = output_img.get_fdata()
    print(output_data.shape)
    # Find all unique classes (labels) in the segmentation
    unique_labels = np.unique(output_data)
    print("Unique labels in the segmentation:", unique_labels)
    
    cardio_ids = list(range(51, 68))
    
    print(cardio_ids)
    
    # select only cardio classes
    cardio_data = np.zeros(output_data.shape)
    for i in cardio_ids:
        cardio_data[output_data == i] = i
    
    # segmentations = nib.load(os.path.join(output_path, "segs2", "segmentations_test.nii.gz"))
    
    # # Create a new NIfTI image from the modified data
    new_nifti = nib.Nifti1Image(cardio_data, output_img.affine)

    # Save the new NIfTI image
    nib.save(new_nifti, f'{output_path}/cardio_segs.nii.gz')

