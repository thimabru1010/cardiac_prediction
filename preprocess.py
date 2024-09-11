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

def sort_dicom_files(dicom_files, patient_path):
    # Read the Instance Number for each DICOM file and store in a list with the file name
    files_with_instance_numbers = []
    for file in dicom_files:
        exam_path = os.path.join(patient_path, file)
        ds = pydicom.dcmread(exam_path)
        instance_number = ds.InstanceNumber
        files_with_instance_numbers.append((file, instance_number))
    
    # Sort the list by Instance Number
    sorted_files = sorted(files_with_instance_numbers, key=lambda x: x[1])
    
    # Extract the sorted file names
    sorted_file_names = [file[0] for file in sorted_files]
    
    return sorted_file_names

def get_affine_matrix(ds):
    # Ler o arquivo DICOM
    # ds = pydicom.dcmread(dicom_file)

    # Obter a orientação do paciente (direção dos eixos x, y, e z)
    orientation = ds.ImageOrientationPatient
    x_dir = orientation[0:3]
    y_dir = orientation[3:6]
    z_dir = [0, 0, 0]  # Será calculado como o produto cruzado de x_dir e y_dir

    # Calcular a direção z (normal ao plano x-y)
    z_dir[0] = x_dir[1] * y_dir[2] - x_dir[2] * y_dir[1]
    z_dir[1] = x_dir[2] * y_dir[0] - x_dir[0] * y_dir[2]
    z_dir[2] = x_dir[0] * y_dir[1] - x_dir[1] * y_dir[0]

    # Obter o espaçamento dos pixels
    px_spacing = ds.PixelSpacing
    slice_thickness = ds.SliceThickness

    # Obter a posição do primeiro slice
    pos = ds.ImagePositionPatient

    # Construir a matriz de afinação
    affine = [
        [x_dir[0] * px_spacing[0], y_dir[0] * px_spacing[1], z_dir[0] * slice_thickness, pos[0]],
        [x_dir[1] * px_spacing[0], y_dir[1] * px_spacing[1], z_dir[1] * slice_thickness, pos[1]],
        [x_dir[2] * px_spacing[0], y_dir[2] * px_spacing[1], z_dir[2] * slice_thickness, pos[2]],
        [0, 0, 0, 1]
    ]

    return affine


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
    # remove controls folder
    patients.remove('Controls')
    print(patients)
    for patient in patients:
        # print(patient)
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
            # print(exam_name)
            exam_name = sort_dicom_files(exam_name, patient_path)
            # print(exam_name)
            for exam_name in tqdm(exam_name):
                exam_path = os.path.join(patient_path, exam_name)
                exam_data = pydicom.dcmread(exam_path)
                output_exam_folder_dicom = os.path.join(output_patient_path, 'DICOM', f'{patient}_{i}')
                create_directory(output_exam_folder_dicom)
                output_exam_dicom = os.path.join(output_exam_folder_dicom, exam_name)
                exam_data.save_as(output_exam_dicom)
                
                # print(exam_data.pixel_array.shape)
                array_data = exam_data.pixel_array
                array_data = np.rot90(exam_data.pixel_array, k=-1)
                array_data = np.flipud(array_data)
                # slices.append(array_data)
                slices.append(array_data)
                # slices.append(exam_data.pixel_array)
            
            output_exam_folder_nifti = os.path.join(output_patient_path, 'Nifti', f'{patient}_{i}')
            create_directory(output_exam_folder_nifti)
            print(output_exam_folder_dicom)
            print(output_exam_folder_nifti)
            dicom2nifti.convert_directory(output_exam_folder_dicom, output_exam_folder_nifti)
            # 1/0
            # Criar um objeto NIfTI
            image_data = np.stack(slices, axis=2)
            print(image_data.shape)
            # affine = np.eye(4)  # Uma matriz de afinação simples, pode precisar ajustar
            affine = get_affine_matrix(exam_data)
            img = nib.Nifti1Image(image_data, affine)
            # Salvar a imagem NIfTI
            nib.save(img, os.path.join(output_exam_folder_nifti, f'{patient}_{i}.nii.gz'))
    
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
