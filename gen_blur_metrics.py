import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from skimage import restoration, io, color
from skimage.measure import blur_effect
import pandas as pd
from utils import get_basename
from gen_fake_gated import var_sharp_score

if __name__ == '__main__':
    root_path = 'data/EXAMES/Exames_NIFTI'
    patients = os.listdir(root_path)
    
    # heart_segs = nib.load('data/EXAMES/Exames_NIFTI')
    # exam_type = 'partes_moles'
    new_img_size = (512, 512)
    # Definir o tamanho inicial do kernel
    psf_size = (15, 15)
    init_psf = np.ones(psf_size)
    blur_metrics = []
    exclude_files = ['multi_label', 'multi_lesion', 'binary_lesion', 'cardiac_CalciumCandidates',\
            'cardiac_CircleSingleLesions', 'cardiac_ROISingleLesions', 'cardiac_circle_lesions', 'cardiac_IncreasedLesion',\
                'cardiac_clustered=', 'cardiac_LesionSingleLesions', 'cardiac_circle']
    keywords_cardiac = ['cardiac']
    for patient in tqdm(patients):
        print(patient)
        
        gated_exam_basename = get_basename(os.listdir(f'{root_path}/{patient}/{patient}'), exclude_files, keywords_cardiac)
        print(gated_exam_basename)
        gated_exam_path = f'{root_path}/{patient}/{patient}/{gated_exam_basename}'
        gated_exam_img = nib.load(gated_exam_path)
        gated_exam = gated_exam_img.get_fdata()
        print(gated_exam.shape)
        
        for i in range(gated_exam.shape[2]):
            # Blind Deconvolution Deblurring usando o método de Richardson-Lucy
            # deblurred, _ = restoration.unsupervised_wiener(ct_data[:, :, i], init_psf)
            blur_m = blur_effect(gated_exam[:, :, i], h_size=11)
            sharp_score = var_sharp_score(gated_exam[:, :, i])
            blur_metrics.append([patient, sharp_score, blur_m, i])
        
        df = pd.DataFrame(blur_metrics, columns=['Patient', 'sharp_score', 'blur_metric', 'Channel'])
        df.to_csv('data/EXAMES/Blur_measures/Gated/blur_metrics.csv', index=False)