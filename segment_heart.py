import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti

if __name__ == '__main__':
    #TODO: Aplicar a segmentação em cada um dos niftis
    input_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/61113/Nifti/61113_0/5_partes_moles__10.nii.gz'
    output_path = 'data/EXAMES_ESCORE_CALCIO_MEDISCAN-20240708T230718Z-001/EXAMES_ESCORE_CALCIO_MEDISCAN/Exames_Separados/61113/Nifti/61113_0'
    # input_img = nib.load(input_path)#.get_fdata()
    # print(input_img.shape)
    #for i in range(input_img.shape[2]):
    # output_img = totalsegmentator(input_img.slicer[:, :, i:i+1], ml=False, task='total')
    # os.system(f'TotalSegmentator -i {input_path} -o {os.path.join(output_path, "segs2", "segmentations_test.nii.gz")} -t total --ml')
    # print(output_img.shape)
    # print(output_img.get_fdata().shape)
    # nib.save(output_img, os.path.join(output_path, 'segs2', f'seg_{i}.nii.gz'))
    # totalsegmentator(input_path, os.path.join(output_path, 'segmentations_test'), ml=True, task='total')
    
    # image_path = os.path.join(output_path, 'total.nii.gz')
    # segmentation_nifti_img, label_map_dict = load_multilabel_nifti(image_path)
    # print(segmentation_nifti_img.shape)
    # print(label_map_dict)
    
    segmentations = nib.load(os.path.join(output_path, "segs2", "segmentations_test.nii.gz"))
    
    # Access the full data array
    segmentation_data = segmentations.get_fdata()

    print(segmentation_data.shape)
    print(segmentations.shape)
    
    # Find all unique classes (labels) in the segmentation
    unique_labels = np.unique(segmentation_data)
    print("Unique labels in the segmentation:", unique_labels)
    
    # Example: Extract and view only the label '1' across all slices
    label_1_data = np.where(segmentation_data == 1, 1, 0)
    print("Shape of the extracted label '1' data:", label_1_data.shape)

    # If you want to visualize this, you can use matplotlib
    plt.imshow(label_1_data[:, :, 50], cmap='gray')  # Display the 51st slice
    plt.show()
    
    # Create a new NIfTI image from the modified data
    new_img = nib.Nifti1Image(label_1_data, segmentations.affine)

    # Save the new NIfTI image
    nib.save(new_img, 'path_to_save/new_segmentation.nii.gz')

