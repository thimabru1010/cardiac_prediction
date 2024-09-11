import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti
import pydicom
    
if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_DICOM'
    # output_path = 'data/EXAMES/Exames_Separados/11517/11517'
    
    pacients = os.listdir(root_path)
    for pacient in tqdm(pacients):
        pacient_path = os.path.join(root_path, pacient)
        output_path = os.path.join('data/EXAMES/Exames_NIFTI', pacient, pacient)
        os.makedirs(output_path, exist_ok=True)
        pacient_path = os.path.join(root_path, pacient, pacient)
        try:
            dicom2nifti.convert_directory(pacient_path, output_path)
        except Exception as e:
                print(f'Error in {pacient}')
                print(e)
    print('finished')
    #TODO: Aplicar a segmentação em cada um dos niftis
    # input_path = '/home/thiago/IDOR/Health_Total_Body_Data/manifest-1690389403229/Healthy-Total-Body-CTs/Healthy-Total-Body-CTs-007/03-17-2001-NA-CTSoft512x512 90min-94809/nifti/205_ctsoft_512x512_90min.nii.gz'
    # # input_img = nib.load(input_path)
    # totalsegmentator(input_path, 'data/output_test4', task='body')
