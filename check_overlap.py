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

def get_dicom_spacing_info(dicom_folder):
    """
    Extracts Pixel Spacing, Slice Thickness, and Spacing Between Slices
    from a DICOM series.

    Parameters:
        dicom_folder (str): Path to the folder containing DICOM slices.

    Returns:
        dict: Containing pixel spacing, slice thickness, and computed slice spacing.
    """
    dicom_files = os.listdir(dicom_folder)
    # Filter only SG files
    dicom_files = [file for file in dicom_files if 'SG' in file]
    # print(dicom_files)
    dicom_files = [os.path.join(dicom_folder, f) for f in dicom_files if f.endswith(".dcm")]
    dicom_slices = [pydicom.dcmread(f) for f in dicom_files]

    # Sort slices by their ImagePositionPatient (z-axis)
    dicom_slices.sort(key=lambda d: d.ImagePositionPatient[2])

    # Get pixel spacing (same for all slices)
    pixel_spacing = dicom_slices[0].PixelSpacing if "PixelSpacing" in dicom_slices[0] else None

    # Get slice thickness (same for all slices)
    slice_thickness = dicom_slices[0].SliceThickness if "SliceThickness" in dicom_slices[0] else None

    # Get SpacingBetweenSlices (may be missing)
    spacing_between_slices = dicom_slices[0].SpacingBetweenSlices if "SpacingBetweenSlices" in dicom_slices[0] else None

    # Compute actual slice-to-slice spacing using ImagePositionPatient
    slice_positions = np.array([ds.ImagePositionPatient[2] for ds in dicom_slices])
    computed_spacing = np.diff(slice_positions)  # Differences between consecutive slice positions

    # If SpacingBetweenSlices is missing, use computed spacing
    actual_spacing = np.mean(computed_spacing) if computed_spacing.size > 0 else None

    return {
        "PixelSpacing (x, y)": pixel_spacing,
        "SliceThickness (z)": slice_thickness,
        "SpacingBetweenSlices (from metadata)": spacing_between_slices,
        "Computed Slice Spacing (from position)": actual_spacing
    }

# # Example usage
# dicom_folder = "path_to_dicom_folder"  # Replace with the actual path
# spacing_info = get_dicom_spacing_info(dicom_folder)

# # Print results
# print("DICOM Spacing Information:")
# for key, value in spacing_info.items():
#     print(f"{key}: {value}")


if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_DICOM'
    # output_path = 'data/EXAMES/Exames_Separados/11517/11517'
    
    patients = os.listdir(root_path)
    patients_error = []
    for patient in tqdm(patients):
        print(patient)
        output_path = os.path.join('data/EXAMES/Exames_NIFTI', patient, patient)
        os.makedirs(output_path, exist_ok=True)
        patient_path = os.path.join(root_path, patient, patient)
        
        # Load DICOM file
        info = get_dicom_spacing_info(patient_path)
        print(info)
        #! Effective Thickness=min(SliceThickness,SpacingBetweenSlices)
        
        # dicom_files = os.listdir(patient_path)
        # dicom_files = [os.path.join(patient_path, file) for file in dicom_files]
        # # filter only SG files
        # dicom_files = [file for file in dicom_files if 'SG' in file]
        # dicom_files = sorted(dicom_files)
        # dicom_slices = [pydicom.dcmread(file) for file in dicom_files]
        # # dicom_slices = [file.pixel_array for file in dicom_files]
        # # dicom_files = np.array(dicom_files)
        # # dicom_files = dicom_files.transpose(1, 2, 0)
        
        # # print pixel spacing, slice thickness and space between slices
        # print(dicom_files[0].PixelSpacing)
        # print(dicom_files[0].SliceThickness)
        # print(dicom_files[1].SliceLocation - dicom_files[0].SliceLocation)
        # break
        
        