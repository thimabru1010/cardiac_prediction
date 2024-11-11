import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

def circumscribing_rectangle(center, radius):
    x_center, y_center = center
    # diameter = 2 * radius
    
    # Coordinates of the rectangle
    x_min = x_center - radius
    x_max = x_center + radius
    y_min = y_center - radius
    y_max = y_center + radius
    
    # Return x, y, w, h
    return x_min, y_min, x_max - x_min, y_max - y_min

def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    patients = os.listdir(root_path)
    
    # heart_segs = nib.load('data/EXAMES/Exames_NIFTI')
    exam_type = 'partes_moles'
    for patient in tqdm(patients):
        print(patient)
        output_path = f'data/EXAMES/Exames_NIFTI/{patient}/{patient}'
        patient_path = os.path.join(root_path, patient, patient)
        # heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_HeartSegs.nii.gz')
        heart_segs_data = nib.load(f'data/EXAMES/Exames_NIFTI/{patient}/{patient}/partes_moles_HeartSegs.nii.gz')
        
        heart_mask = heart_segs_data.get_fdata()
        if heart_mask[heart_mask != 0].shape[0] == 0:
            print(f'No heart mask found in {patient}')
            continue
        heart_mask[heart_mask != 0] = 1
        
        # dilation of the mask
        kernel = np.ones((10,10), np.uint8)
        cardio_data = cv2.dilate(heart_mask, kernel, iterations=3)
        
        # Get the slice with the maximum area
        # count_area = np.zeros((1, 1, cardio_data.shape[2]))
        count_area = np.sum(cardio_data, axis=(0, 1))
        # print(count_area)
        max_index = np.argmax(count_area)
        max_slice = cardio_data[:, :, max_index]
        coordinates = np.argwhere(max_slice)
        # print(coordinates)

        # Calculate the centroid from the Total Segmentator Heart Mask
        centroid = coordinates.mean(axis=0)
        # print(centroid)
        
        circle_mask = np.zeros(max_slice.shape)
        radius = 150
        center = (int(centroid[1]), int(centroid[0]))
        cv2.circle(circle_mask, center, radius=radius, color=1, thickness=-1)
        # Get the rectangle circunscribing the circle
        # Get the coordinates of the non-zero pixels in the circle mask
        non_zero_coords = np.argwhere(circle_mask)
        rect = cv2.boundingRect(non_zero_coords)
        
        rect = circumscribing_rectangle(center, radius)
        x, y, w, h = rect
        
        # repeate circle mask for all slices
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], cardio_data.shape[2], axis=2)
        # print(circle_mask.shape)

        # Save the new NIfTI image
        new_nifti = nib.Nifti1Image(circle_mask, heart_segs_data.affine)
        nib.save(new_nifti, f'{output_path}/partes_moles_FakeGated_CircleMask.nii.gz')
        
        nifti_files = os.listdir(patient_path)
        motion_filename = get_partes_moles_basename(nifti_files)
        # try:
        #     # motion_filename = [file for file in nifti_files if 'partes_moles_body' in file][0]
        #     motion_filename = get_partes_moles_basename(nifti_files)
        # except:
        #     print(f'partes_moles_body not found!: {patient}')
        #     break
        
        input_img = nib.load(os.path.join(patient_path, motion_filename))#.get_fdata()
        ct_data = input_img.get_fdata()
        
        min_value = ct_data.min()
        ct_data = ct_data * circle_mask
        ct_data[circle_mask == 0] = min_value
        # print(ct_data.max())
        
        ct_data = ct_data[y:y+h, x:x+w]
        # print(ct_data.shape)
        
        img_size = (512, 512)
        ct_data2 = np.zeros((img_size[0], img_size[1], ct_data.shape[2]))
        for i in range(ct_data.shape[2]):
            #TODO: Apply a filter to remove motion blur
            ct_data2[:, :, i] = cv2.resize(ct_data[:, :, i], img_size, interpolation=cv2.INTER_LANCZOS4)
        
        # print(ct_data2.shape)
        
        # # Create a new NIfTI image from the modified data
        new_nifti = nib.Nifti1Image(ct_data2, heart_segs_data.affine)

        # Save the new NIfTI image
        nib.save(new_nifti, f'{output_path}/partes_moles_FakeGated.nii.gz')
        # break