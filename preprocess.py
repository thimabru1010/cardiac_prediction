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

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def convert_dicom_to_nifti(patients, root_path):
    for patient in tqdm(patients):
        patient_path = os.path.join(root_path, patient)
        pat_exam = os.listdir(patient_path)[0]
        exam_path = os.path.join(patient_path, pat_exam)
        subfolder_exam_path = os.path.join(exam_path, os.listdir(exam_path)[0])
        output_path = os.path.join('/home/thiago/IDOR/Health_Total_Body_Data/Nifti_files', patient)
        create_directory(output_path)
        # Check if any file is inside output_path
        if os.listdir(output_path):
            continue
        dicom2nifti.convert_directory(subfolder_exam_path, output_path)
        
def separate_patient_exams(root_path):
    patients = os.listdir(root_path)
    for patient in patients:
        patient_path = os.path.join(root_path, patient)
        exams = os.listdir(patient_path)
        for exam in exams:
            exam_path = os.path.join(patient_path, exam)
            subfolder_exam_path = os.path.join(exam_path, os.listdir(exam_path)[0])
            output_path = os.path.join('/home/thiago/IDOR/Health_Total_Body_Data/Nifti_files', patient)
            create_directory(output_path)
            # Check if any file is inside output_path
            if os.listdir(output_path):
                continue
            dicom2nifti.convert_directory(subfolder_exam_path, output_path)
    
if __name__ == '__main__':
    root_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Todos'
    output_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados'
    
    separate_patient_exams(root_path)
    
    patients = os.listdir(root_path)
    create_directory('/home/thiago/IDOR/Health_Total_Body_Data/Nifti_files')
    convert_dicom_to_nifti(patients, root_path)
    
    #TODO: Aplicar a segmentação em cada um dos niftis
    # input_path = '/home/thiago/IDOR/Health_Total_Body_Data/manifest-1690389403229/Healthy-Total-Body-CTs/Healthy-Total-Body-CTs-007/03-17-2001-NA-CTSoft512x512 90min-94809/nifti/205_ctsoft_512x512_90min.nii.gz'
    # # input_img = nib.load(input_path)
    # totalsegmentator(input_path, 'data/output_test4', task='body')
