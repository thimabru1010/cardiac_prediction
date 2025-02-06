import pydicom
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
from numpy import linalg as LA
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti
import pandas as pd
import seaborn as sns
from calculate_score import get_basename
from scipy.spatial.distance import cdist

if __name__=='__main__':
    
    csv_path = 'data/EXAMES/classifier_dataset_radius=150_clustered=5.csv'
    df = pd.read_csv(csv_path)
    n_clusters = csv_path.split('=')[-1].split('.')[0]
    
    df[['Centroid X', 'Centroid Y', 'Channel']] = df[['Centroid X', 'Centroid Y', 'Channel']].astype(int)
    
    # data_pacient = df.groupby('Pacient')
    
    pacients = df['Pacient'].unique()
    root_path = 'data/EXAMES/Exames_NIFTI'
    for pacient in pacients:
        print(pacient)
        # Load Nifiti for the selected pacient
        gated_exam_basename = get_basename(os.listdir(f'{root_path}/{pacient}/{pacient}'))
        gated_exam_path = f'{root_path}/{pacient}/{pacient}/{gated_exam_basename}'
        gated_exam_img = nib.load(gated_exam_path)#.get_fdata()
        gated_exam = gated_exam_img.get_fdata()
        print(gated_exam.shape)
        
        circle_mask = np.zeros(gated_exam.shape[:2])
        circle_mask = cv2.circle(circle_mask, (circle_mask.shape[1] // 2, circle_mask.shape[0] // 2), 120, 1, -1)
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], gated_exam.shape[2], axis=2)
        
        clusters_mask = np.zeros_like(gated_exam)
        print(clusters_mask.shape)
        
        # print(df.columns)
        points = df[df['Pacient'] == pacient]
        print(points.shape)
        
        clusters_mask[points['Centroid X'].values, points['Centroid Y'].values,\
            points['Channel'].values] = points['Cluster'].values
        
        calcium_candidates = gated_exam.copy()
        calcium_candidates[calcium_candidates < 130] = 0
        calcium_candidates[calcium_candidates >= 130] = 1
        
        calcium_candidates = calcium_candidates * circle_mask
        
        coordinates = np.argwhere(calcium_candidates)
        calcium_centroid = np.mean(coordinates, axis=0)
        
        conected_lesions = np.zeros_like(gated_exam)
        for channel in range(gated_exam.shape[2]):
            _, lesions = cv2.connectedComponents(calcium_candidates[:, :, channel].astype(np.uint8))
            conected_lesions[:, :, channel] = lesions
            lesion_labels = list(np.unique(lesions))
            # print(lesions.shape, lesion_labels)
            points_channel = points[points['Channel'] == channel].copy()
            if points_channel.shape[0] == 0:
                    continue
                
            if 0 in lesion_labels:
                lesion_labels.remove(0)
                
            for lesion_label in lesion_labels:
                lesion = lesions.copy()
                lesion[lesion != lesion_label] = 0
                lesion[lesion == lesion_label] = 1
                coordinates = np.argwhere(lesion)
                centroid = np.mean(coordinates, axis=0)
                # print(centroid)
                # centroid[2] = channel
                
                # Calculate distances
                points_channel['distances'] = points_channel[['Centroid X', 'Centroid Y']].apply(
                    lambda row: LA.norm(row.values - centroid), axis=1)
                min_distance = points_channel['distances'].min()
                # print(min_distance)
                
                min_dist_cluster = points_channel[points_channel['distances'] == min_distance]['Cluster'].values[0]
                # lesion[lesion == lesion_label] = min_dist_cluster
                calcium_candidates[lesion != 0, channel] = min_dist_cluster
                
        # Save the clustered image
        clustered_img = nib.Nifti1Image(calcium_candidates, affine=gated_exam_img.affine)
        nib.save(clustered_img, f'{root_path}/{pacient}/{pacient}/cardiac_clustered={n_clusters}.nii.gz')