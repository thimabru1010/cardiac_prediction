import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom, affine_transform
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift  # or use cv2.warpAffine for integer shift
import os
from masks_auto_generation.extract_text_from_image import extract_text_from_image
from masks_auto_generation.remove_text_from_image import remove_text_from_image
import google.generativeai as genai # type: ignore
from PIL import Image
import re
from tqdm import tqdm
from masks_auto_generation.gen_seg_mask import load_dicoms, window_level, artifficial_zoom_crop, tight_crop,\
    align_mask_to_ct, hue_mask
from utils import create_save_nifti, save_slice_as_dicom
from openai import OpenAI
from PIL import Image
import pandas as pd
    
if __name__ == "__main__":
    root_path = 'data/ExamesArya'
    root_output = 'data/ExamesArya_TextInfo'
    debug_folder = 'data/Debug'
    os.makedirs(debug_folder, exist_ok=True)
    os.makedirs(root_output, exist_ok=True)
    # root_path = 'data/EXAMES/Exames_DICOM'
    # output_path = 'data/EXAMES/Exames_Separados/11517/11517'
    
    # Resize cropped mask to 512×512 for plotting
    target_size = (512, 512)
    kernel = np.ones((3,3), np.uint8)
    
    # Color ranges in HSV space
    RED1  = (0,   5)      # 0°-10°
    RED2  = (170, 179)    # 340°-360°
    GREEN = (50,  80)     # 100°-160°  (LAD in your screenshots)
    BLUE  = (100,130)     # 200°-260°  (CX)
    PINK  = (145,175)     # 290°-350°  (calcification contour)
    
    patients = os.listdir(root_path)
    patients_error = []
    
    # Filter patients already processed
    patients_processed = os.listdir(root_output)
    # Filter only folders
    patients_processed = [p for p in patients_processed if os.path.isdir(os.path.join(root_output, p))]
    patients = [p for p in patients if p not in patients_processed]
    print(f"Patients already processed: {len(patients_processed)}")
    print(f"Patients to process: {len(patients)}")
    
    if len(patients_processed) > 0:
        # Load csv with already processed slices
        df_existing = pd.read_csv(os.path.join(root_output, 'slices_text_info.csv'))
    
    client = OpenAI()
    
    total_slice_positions = []
    total_zoom_values = []
    total_patient_ids = []
    total_number_channels = []
    total_mask_slices_channels = []
    # Load IA.dcm images and generates mask labels from it
    for patient in tqdm(patients):
        print("\n\nPreprocessing patient:", patient)
        patient_output_folder = os.path.join(root_output, patient)
        os.makedirs(patient_output_folder, exist_ok=True)
        # load dicom images
        patient_path = os.path.join(root_path, patient)
        ct_img, mask_img = load_dicoms(patient_path)
        mask_np = sitk.GetArrayFromImage(mask_img)
        ct_np = sitk.GetArrayFromImage(ct_img)
        print("Mask image shape:", mask_np.shape)
        print("CT image shape:", ct_np.shape)
        slice_positions = []
        zoom_values = []
        patient_ids = []
        number_channels = []
        mask_slices_channels = []
    
        calc_masks = []
        ct_exams = []
        for i in range(mask_np.shape[0]):
            mask_slice = mask_np[i]
            
            # zoom_factor, slice_position = extract_text_from_image(
            #     model_name='gemini-2.0-flash-thinking-exp-01-21',
            #     prompt="Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'.",
            #     image_pil=mask_slice) # type: ignore
            
            zoom_factor, slice_position = extract_text_from_image(
                client=client,
                model_name='gpt-4.1',
                prompt="Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'.",
                image_npy=mask_slice) # type: ignore

            print(f"Patient: {patient} - Zoom factor: {zoom_factor} - Slice position: {slice_position}")

            # Store zoom and slice position for each patient
            slice_positions.append(slice_position)
            zoom_values.append(zoom_factor)
            patient_ids.append(patient)
            number_channels.append(ct_np.shape[0])
            mask_slices_channels.append(i)

            total_slice_positions.append(slice_position)
            total_zoom_values.append(zoom_factor)
            total_patient_ids.append(patient)
            total_number_channels.append(ct_np.shape[0])
            total_mask_slices_channels.append(i)
            
            
        # Save zoom and slice in a csv file
        df = pd.DataFrame({
            'patient_id': patient_ids,
            'mask_slice_channel': mask_slices_channels,
            'slice_position': slice_positions,
            'zoom_factor': zoom_values,
            'ct_number_channels': number_channels
        })

        df.to_csv(os.path.join(patient_output_folder, f'{patient}_slices_text_info.csv'), index=False)

        print(f"Saved text info CSV for patient {patient} at {os.path.join(patient_output_folder, f'{patient}_slices_text_info.csv')}")
    # Save zoom and slice in a csv file
    df = pd.DataFrame({
        'patient_id': total_patient_ids,
        'mask_slice_channel': total_mask_slices_channels,
        'slice_position': total_slice_positions,
        'zoom_factor': total_zoom_values,
        'ct_number_channels': total_number_channels
    })
    
    if len(patients_processed) > 0:
        print(f"Adding new {df.shape[0]} slices info to existing {df_existing.shape[0]} slices info.")
        # Concatenate existing dataframe with new data
        df = pd.concat([df_existing, df], ignore_index=True)
        print(f"Total slices info after concatenation: {df.shape[0]}")
    
    df.to_csv(os.path.join(root_output, 'slices_text_info.csv'), index=False)