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
from gen_seg_mask import load_dicoms, window_level, artifficial_zoom_crop, tight_crop,\
    align_mask_to_ct, hue_mask
from utils import create_save_nifti, save_slice_as_dicom
from openai import OpenAI

def save_as_nifti(array: np.ndarray, output_path: str, spacing=(1.0, 1.0, 1.0)):
    """
    Salva um numpy array como arquivo NIFTI.
    
    Parameters:
    -----------
    array : np.ndarray
        Array 3D com shape (Z, H, W)
    output_path : str
        Caminho completo do arquivo .nii ou .nii.gz
    spacing : tuple
        Espaçamento entre voxels (z, y, x)
    """
    # Converter para SimpleITK Image
    sitk_img = sitk.GetImageFromArray(array)
    sitk_img.SetSpacing(spacing)
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar
    sitk.WriteImage(sitk_img, output_path)
    print(f"Saved: {output_path}")
    
if __name__ == "__main__":
    root_path = 'data/ExamesArya'
    root_output = 'data/ExamesArya_NIFTI_CalcSegTraining'
    debug_folder = 'data/Debug'
    os.makedirs(debug_folder, exist_ok=True)
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
    patients = [p for p in patients if p not in patients_processed]
    print(f"Patients already processed: {len(patients_processed)}")
    print(f"Patients to process: {len(patients)}")
    
    client = OpenAI()
    # Load IA.dcm images and generates mask labels from it
    for patient in tqdm(patients):
        print("\n\nPreprocessing patient:", patient)
        # load dicom images
        patient_path = os.path.join(root_path, patient)
        ct_img, mask_img = load_dicoms(patient_path)
        mask_np = sitk.GetArrayFromImage(mask_img)
        ct_np = sitk.GetArrayFromImage(ct_img)
        print("Mask image shape:", mask_np.shape)
        print("CT image shape:", ct_np.shape)
        slice_positions = []
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
            
            print(f"Zoom factor: {zoom_factor} - Slice position: {slice_position}")
            mask_slice, _ = remove_text_from_image(mask_slice)  # Remove text from mask
            
            print(f"Slice position: {slice_position}")
            ct_slice = ct_np[slice_position-1]  #! O indice 0 é o final da série
            ct_slice = window_level(ct_slice)       # Use first slice, apply window/level
            print(ct_slice.shape, ct_slice.min(), ct_slice.max())
            print(mask_slice.shape, mask_slice.min(), mask_slice.max())
            
            # Apply manual crop (zoom effect, no resize)
            print(mask_np.shape)
            cropped_mask_slice = artifficial_zoom_crop(mask_slice, zoom_factor)
            cropped_mask_slice = tight_crop(cropped_mask_slice, thr=10)
            ct_slice = tight_crop(ct_slice, thr=0)
            print(cropped_mask_slice.shape, cropped_mask_slice.min(), cropped_mask_slice.max())
            print(ct_slice.shape, ct_slice.min(), ct_slice.max())

            # Resize both exams to target_size
            cropped_mask_resized = cv2.resize(cropped_mask_slice, target_size, interpolation=cv2.INTER_NEAREST_EXACT)
            ct_slice = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            cropped_mask_resized = cv2.morphologyEx(cropped_mask_resized, cv2.MORPH_OPEN, kernel)
            # Save cropped_mask_resize and ct_slice for debugging
            print("Debug info after cropping and resizing:")
            print(cropped_mask_resized.shape, cropped_mask_resized.min(), cropped_mask_resized.max())
            print(ct_slice.shape, ct_slice.min(), ct_slice.max())
            debug_patient_folder = os.path.join(debug_folder, patient)
            os.makedirs(debug_patient_folder, exist_ok=True)
            create_save_nifti(cropped_mask_resized, np.eye(4), os.path.join(debug_patient_folder, f"{patient}_slice{slice_position:03d}_mask.nii.gz"))
            create_save_nifti(ct_slice, np.eye(4), os.path.join(debug_patient_folder, f"{patient}_slice{slice_position:03d}_ct.nii.gz"))
            
            cropped_mask_resized = align_mask_to_ct(cropped_mask_resized, ct_slice, return_warp=True)

            print(cropped_mask_resized.shape, cropped_mask_resized.min(), cropped_mask_resized.max())
            print(ct_slice.shape, ct_slice.min(), ct_slice.max())
            
            calc_candidates = ct_slice.copy()
            calc_candidates[ct_slice < 130] = 0
            calc_candidates[ct_slice > 130] = 1
            
            print(cropped_mask_resized.shape)
            # calc_candidates = np.stack([
            #     calc_candidates,
            #     calc_candidates,
            #     calc_candidates
            # ], axis=-1)

            print(cropped_mask_resized.shape, cropped_mask_resized.min(), cropped_mask_resized.max())
            print(ct_slice.shape, ct_slice.min(), ct_slice.max())
            
            mask_segs = cv2.cvtColor(cropped_mask_resized, cv2.COLOR_RGB2HSV)
            green_mask = hue_mask(mask_segs, GREEN) * calc_candidates
            blue_mask = hue_mask(mask_segs, BLUE) * calc_candidates
            red_mask1 = hue_mask(mask_segs, RED1) * calc_candidates
            red_mask2 = hue_mask(mask_segs, RED2) * calc_candidates
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            pink_mask = hue_mask(mask_segs, PINK) * calc_candidates

            print("Unique values in masks:", np.unique(green_mask), np.unique(blue_mask), np.unique(red_mask), np.unique(pink_mask))

            calc_mask = 1 * green_mask + 2 * blue_mask + 3 * red_mask + 4 * pink_mask

            slice_positions.append(slice_position)
            calc_masks.append(calc_mask)
            ct_exams.append(ct_slice)
            
        # Save the mask slice in NIFTI format
        slice_positions = np.array(slice_positions)
        sort_idx = np.argsort(slice_positions)
        calc_masks = np.array(calc_masks)[sort_idx]
        ct_exams = np.array(ct_exams)[sort_idx]
        print(calc_masks.shape, ct_exams.shape)
        
        save_as_nifti(calc_masks, os.path.join(root_output, patient, f"{patient}_mask.nii.gz"), spacing=ct_img.GetSpacing())
        save_as_nifti(ct_exams, os.path.join(root_output, patient, f"{patient}_gated_prep.nii.gz"), spacing=ct_img.GetSpacing())
        print("Saved patient:", patient)

    print('Finished')