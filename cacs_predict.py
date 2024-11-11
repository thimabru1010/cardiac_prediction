#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 09:41:53 2021

@author: Bernhard Foellmer
"""

import os, sys
import torch
from torch import nn, optim
from MTAL_CACS.src.model import MTALModel
from glob import glob
import SimpleITK as sitk
import numpy as np
import pathlib
import nibabel as nib
import argparse

def get_cardiac_basename(files):
    exclude_files = ['multi_label', 'multi_lesion', 'binary_lesion']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'cardiac' in file]
    return gated_exam_basename[0]

def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated', 'partes_moles_FakeGated_CircleMask']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

def main(args):
    print('--- Start processing ---')
    # Define directories
    data_dir = args.data_dir
    model_dir = args.model_dir
    prediction_dir = args.prediction_dir

    # Create directory
    os.makedirs(prediction_dir, exist_ok=True)

    # Create model
    model = MTALModel(args.device)
    # Initialize model
    model.create()
    # Load pretrained model parameter
    print('Loading model: ' + model_dir)
    model.load(model_dir)
    
    # model.eval()

    # Load image files from data folder    
    # files = glob(data_dir + '/*.nii.gz')
    patients = os.listdir(data_dir)
    for patient in patients:
        print('Processing patient: ' + patient)
        # Read image
        basename = get_gated_cardiac_basename(os.listdir(os.path.join(data_dir, patient, patient)))
        if args.partes_moles:
            print('Inferring partes_moles')
            basename = get_partes_moles_basename(os.listdir(os.path.join(data_dir, patient, patient)))
        print(basename)
        filename = os.path.splitext(basename)[0]
        # image_nifti = nib.load(os.path.join(data_dir, patient, patient, basename))
        image_sitk = sitk.ReadImage(os.path.join(data_dir, patient, patient, basename))
        image = sitk.GetArrayFromImage(image_sitk)
        # image = image_nifti.get_fdata()
        # image = np.transpose(image, (2, 1, 0))
        # print(image.shape)

        # Normalize image data
        Xmin = -2000
        Xmax = 1300
        image[image==-3024]=-2048
        image_norm = (image - Xmin) / (Xmax - Xmin)
        
        # Lesion candidate mask
        lesion_candidate_mask = np.zeros(image.shape)
        lesion_candidate_mask[image>130] = 1
        
        # Init predictions
        pred_lesion = np.zeros(image.shape)
        pred_lesion_multi = np.zeros(image.shape)
        pred_region = np.zeros(image.shape)
        
        # print('Predicting CT: ' + os.path.basename(file))
        # Iterate over slices
        for s in range(image.shape[0]):

            # Convert to torch tensor
            Ximage = torch.FloatTensor(image_norm[s:s+1]).to('cuda')
            Xmask = torch.FloatTensor(lesion_candidate_mask[s:s+1]).to('cuda')
            Xin = torch.cat((Ximage, Xmask), dim=0).unsqueeze(0)
            
            # Predict CT slice
            Y_region, Y_lesion = model.predict(Xin)
                
            # Combine lesion predictions
            Y_lesion_bin = Y_lesion.round()
            Y_lesion_bin[:,1,:,:] = Y_lesion_bin[:,1,:,:]*Xmask
            Y_lesion_bin[:,0,:,:] = 1-Y_lesion_bin[:,1,:,:]
            
            # Combine region predictions
            Y_region_bin = torch.zeros(Y_region.shape, dtype=torch.float32).to('cuda')
            Y_region_bin = Y_region_bin.scatter_(1,torch.argmax(Y_region, dim=1, keepdim=True) , 1)
            Y_region_multi = torch.argmax(Y_region_bin, dim=1, keepdim=True)
            
            # Combine lesion and region predictions
            Y_lesion_multi = torch.argmax(torch.cat((torch.max(Y_lesion_bin[:,0:1,:,:], Y_region_bin[:,0:1,:,:]), Y_lesion_bin[:,1:2,:,:].repeat((1,3,1,1)) * Y_region_bin[:,1:,:,:]), dim=1), dim=1, keepdim=True)
            
            # Fill predictions
            pred_lesion[s,:,:] = Y_lesion_bin[0,1,:,:].cpu()
            pred_region[s,:,:] = Y_region_multi[0,0,:,:].cpu()
            pred_lesion_multi[s,:,:] = Y_lesion_multi[0,:,:,:].cpu()
            
            # print(pred_lesion_multi[s,:,:])
            # print(np.unique(pred_lesion_multi[s,:,:]))
            
        # print('Saveing predictions from: ' + os.path.basename(file))
        # Save predictions
        # filepath = os.path.join(prediction_dir, filename + '_binary_lesion.nrrd')
        prediction_path = os.path.join(prediction_dir, patient, patient)
        filepath = os.path.join(prediction_path, filename + '_binary_lesion.nii.gz')
        
        # Save predictions as nifti
        # new_nifti = nib.Nifti1Image(pred_lesion, image_nifti.affine)
        # nib.save(new_nifti, filepath)
        
        Y_lesion_bin_sitk = sitk.GetImageFromArray(pred_lesion)
        Y_lesion_bin_sitk.CopyInformation(image_sitk)
        sitk.WriteImage(Y_lesion_bin_sitk, filepath, True)

        # Save predictions
        # filepath = os.path.join(prediction_dir, filename + '_multi_label.nrrd')
        filepath = os.path.join(prediction_path, filename + '_multi_label.nii.gz')
        
        # new_nifti = nib.Nifti1Image(pred_lesion_multi, image_nifti.affine)
        # nib.save(new_nifti, filepath)
        Y_region_sitk = sitk.GetImageFromArray(pred_region)
        Y_region_sitk.CopyInformation(image_sitk)
        sitk.WriteImage(Y_region_sitk, filepath, True)
        
        # filepath = os.path.join(prediction_dir, filename + '_multi_lesion.nrrd')
        filepath = os.path.join(prediction_path, filename + '_multi_lesion.nii.gz')
        
        # new_nifti = nib.Nifti1Image(pred_lesion_multi, image_nifti.affine)
        # nib.save(new_nifti, filepath)
        Y_lesion_multi_sitk = sitk.GetImageFromArray(pred_lesion_multi)
        Y_lesion_multi_sitk.CopyInformation(image_sitk)
        sitk.WriteImage(Y_lesion_multi_sitk, filepath, True)
        
    print('--- Finished processing ---')
        

if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Coronary Calcium Scoring with Multi-Task model'
    )

    parser.add_argument('--model_dir', '-m', type=str,
                        action='store', dest='model_dir',
                        help='Directory of the model')
    parser.add_argument('--data_dir', '-d', type=str,
                        action='store', dest='data_dir',
                        help='Directory of data')
    parser.add_argument('--prediction_dir', '-p', type=str,
                        action='store', dest='prediction_dir',
                        help='Directory of predictions')
    parser.add_argument('--device', '-gpu', type=str,
                        action='store', dest='device',
                        help='Devoce NO. of GPU')
    parser.add_argument('--partes_moles', action='store_true', help='Whether to infer partes_moles exams')

    args = parser.parse_args()
    main(args)
