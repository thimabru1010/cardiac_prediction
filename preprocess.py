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
    
    patients = os.listdir(root_path)
    patients_error = []
    for patient in tqdm(patients):
        patient = '180132'
        print(patient)
        patient_path = os.path.join(root_path, patient)
        output_path = os.path.join('data/EXAMES/Exames_NIFTI', patient, patient)
        os.makedirs(output_path, exist_ok=True)
        patient_path = os.path.join(root_path, patient, patient)
        # dicom2nifti.convert_directory(patient_path, output_path)
        try:
            dicom2nifti.convert_directory(patient_path, output_path)
        except Exception as e:
                print(f'Error in {patient}')
                patients_error.append(patient)
                print(e)
        break
    print('finished')
    
    print('Errors found in patients:')
    print(patients_error)
