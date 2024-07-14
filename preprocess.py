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
        
def separate_patient_exams(root_path, output_path):
    patients = os.listdir(root_path)
    for patient in patients:
        print(patient)
        patient_path = os.path.join(root_path, patient)
        output_patient_path = os.path.join(output_path, patient)
        create_directory(output_patient_path)
        
        exams = os.listdir(patient_path)
        # Split exams names into groups of 3
        exams_0 = []
        exams_1 = []
        exams_2 = []
        exams_all = []
        for exam in exams:
            if '(1)' in exam:
                exams_1.append(exam)
            elif '(2)' in exam:
                exams_2.append(exam)
            else:
                exams_0.append(exam)
        
        # print(len(exams))
        print(len(exams_0), len(exams_1), len(exams_2))
        
        # if len(exams_2) != 0:
        #     for exam_2_name in tqdm(exams_2):
        #         exam_path_2 = os.path.join(patient_path, exam_2_name)
        #         exam_2_data = pydicom.dcmread(exam_path_2)
        #         output_exam_2_folder_dicom = os.path.join(output_patient_path, 'DICOM', f'{patient}_2')
        #         create_directory(output_exam_2_folder_dicom)
        #         output_exam_2_dicom = os.path.join(output_exam_2_folder_dicom, exam_2_name)
        #         exam_2_data.save_as(output_exam_2_dicom)
        
        
        for i, exam_name in enumerate([exams_0, exams_1, exams_2]):
            if len(exam_name) == 0:
                continue
            slices = []
            for exam_name in tqdm(exam_name):
                exam_path = os.path.join(patient_path, exam_name)
                exam_data = pydicom.dcmread(exam_path)
                output_exam_folder_dicom = os.path.join(output_patient_path, 'DICOM', f'{patient}_{i}')
                create_directory(output_exam_folder_dicom)
                output_exam_dicom = os.path.join(output_exam_folder_dicom, exam_name)
                exam_data.save_as(output_exam_dicom)
                
                # print(exam_data.pixel_array.shape)
                
                slices.append(exam_data.pixel_array)
            
            output_exam_folder_nifti = os.path.join(output_patient_path, 'Nifti', f'{patient}_{i}')
            create_directory(output_exam_folder_nifti)
            print(output_exam_folder_dicom, output_exam_folder_nifti)
            dicom2nifti.convert_directory(output_exam_folder_dicom, output_exam_folder_nifti)
            
            # Criar um objeto NIfTI
            image_data = np.stack(slices, axis=2)
            print(image_data.shape)
            affine = np.eye(4)  # Uma matriz de afinação simples, pode precisar ajustar
            img = nib.Nifti1Image(image_data, affine)
            # Salvar a imagem NIfTI
            nib.save(img, os.path.join(output_exam_folder_nifti, f'{patient}_{i}.nii.gz'))
    
if __name__ == '__main__':
    root_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Todos'
    output_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados'
    
    separate_patient_exams(root_path, output_path)
    
    # Check shape of nifti files
    # output_nifti_path = os.path.join(output_path, 'Nifti')
    patients = os.listdir(output_path)
    for patient in patients:
        patient_path = os.path.join(output_path, patient, 'Nifti')
        exams = os.listdir(patient_path)
        for exam in exams:
            exam_path = os.path.join(patient_path, exam)
            nifti_files = os.listdir(exam_path)
            for nifti_file in nifti_files:
                nifti_path = os.path.join(exam_path, nifti_file)
                img = nib.load(nifti_path)
                print(nifti_path)
                print(img.shape)
    
    #TODO: Aplicar a segmentação em cada um dos niftis
    # input_path = '/home/thiago/IDOR/Health_Total_Body_Data/manifest-1690389403229/Healthy-Total-Body-CTs/Healthy-Total-Body-CTs-007/03-17-2001-NA-CTSoft512x512 90min-94809/nifti/205_ctsoft_512x512_90min.nii.gz'
    # # input_img = nib.load(input_path)
    # totalsegmentator(input_path, 'data/output_test4', task='body')
