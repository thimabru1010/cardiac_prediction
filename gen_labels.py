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
from gen_seg_mask import load_dicoms, window_level

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
        ct_img, mask_img = load_dicoms(patient_path)
        mask_np = sitk.GetArrayFromImage(mask_img)
        ct_np = sitk.GetArrayFromImage(ct_img)
        print("Mask image shape:", mask_np.shape)
        print("CT image shape:", ct_np.shape)
        1/0
        for i in range(mask_np.shape[0]):
            mask_slice = mask_np[i]
            
            zoom_factor, slice_position = extract_text_from_image(
                model_name='gemini-2.0-flash-thinking-exp-01-21',
                prompt="Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'.",
                image_pil=Image.fromarray(mask_slice)) # type: ignore
            
            print(f"Zoom factor: {zoom_factor} - Slice position: {slice_position}")
            mask_slice, _ = remove_text_from_image(mask_slice)  # Remove text from mask
            
            # slice_coord = ct_np.shape[0] + 1 - slice_position + 1
            print(f"Slice position: {slice_position}")
            ct_slice = ct_np[slice_position-1]  #! O indice 0 é o final da série
            ct_slice = window_level(ct_slice)       # Use first slice, apply window/level
    
        # label_dicom_files = [f for f in os.listdir(patient_path) if "IA" in f]
        # for dicom_file in label_dicom_files:
        #     dicom_path = os.path.join(patient_path, dicom_file)
        #     ds = pydicom.dcmread(dicom_path)
        #     # process dicom image and generate mask
    print('finished')
    
    print('Errors found in patients:')
    print(patients_error)