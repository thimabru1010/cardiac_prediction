import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom, affine_transform
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift  # or use cv2.warpAffine for integer shift
import os
from extract_text_from_image import extract_text_from_image
from remove_text_from_image import remove_text_from_image
import google.generativeai as genai # type: ignore
from PIL import Image
import re
from tqdm import tqdm

if __name__ == "__main__":
    root_path = 'data/ExamesArya'
    root_output = 'data/ExamesArya_NIFTI'
    # root_path = 'data/EXAMES/Exames_DICOM'
    # output_path = 'data/EXAMES/Exames_Separados/11517/11517'
    
    patients = os.listdir(root_path)
    patients_error = []
    # Load IA.dcm images and generates mask labels from it
    for patient in tqdm(patients):
        # load dicom images
        patient_path = os.path.join(root_path, patient)
        label_dicom_files = [f for f in os.listdir(patient_path) if "IA" in f]
        for dicom_file in label_dicom_files:
            dicom_path = os.path.join(patient_path, dicom_file)
            ds = pydicom.dcmread(dicom_path)
            # process dicom image and generate mask
    print('finished')
    
    print('Errors found in patients:')
    print(patients_error)