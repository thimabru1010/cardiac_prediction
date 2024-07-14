import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator, totalseg_combine_masks
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti
import xmltodict

if __name__ == '__main__':
    input_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/61113/Nifti/61113_0/5_partes_moles__10.nii.gz'
    output_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/61113/Nifti/61113_0'
        
    output_img = nib.load(os.path.join(output_path, 'total.nii.gz'))
    print(output_img.get_fdata().shape)
    
    ext_header = output_img.header.extensions[0].get_content()
    ext_header = xmltodict.parse(ext_header)
    print(ext_header)
    
    heart = output_img.slicer[:, :, 51:69]
    print(heart.shape)
    # heart = heart[:, :, 51:69]
    print(heart.shape)
    
    # affine = np.eye(4)  # This is a simple identity matrix, used for lack of a specific affine
    heart_nifti = nib.Nifti1Image(heart, output_img.affine)
    print(heart_nifti.shape)
    heart_nifti = heart
    
    nib.save(heart_nifti, os.path.join(output_path, 'heart.nii.gz'))
    
    totalseg_combine_masks()