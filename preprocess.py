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
    pacients_error = []
    for pacient in tqdm(pacients):
        print(pacient)
        pacient_path = os.path.join(root_path, pacient)
        output_path = os.path.join('data/EXAMES/Exames_NIFTI', pacient, pacient)
        os.makedirs(output_path, exist_ok=True)
        pacient_path = os.path.join(root_path, pacient, pacient)
        try:
            dicom2nifti.convert_directory(pacient_path, output_path)
        except Exception as e:
                print(f'Error in {pacient}')
                pacients_error.append(pacient)
                print(e)
    print('finished')
    
    print('Errors found in pacients:')
    print(pacients_error)
