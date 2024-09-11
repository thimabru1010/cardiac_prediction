import pydicom
import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
from numpy import linalg as LA
import cv2
from tqdm import tqdm
import argparse
import dicom2nifti
import pandas as pd

if __name__ == '__main__':
    excel_path = 'data/EXAMES/cac_score_data.xlsx'
    df = pd.read_excel(excel_path)
    
    print(df.head())
    ids = df['ID'].tolist()
    
    ids_exams = os.listdir('data/EXAMES/Exames_Todos')
    print(ids_exams)
    
    ids_exams = [174848]
    
    # Select ids in DataFrame that are equal to ids_exams list
    valid_data = df[df['ID'].isin(ids_exams)]
    
    print()
    print(valid_data.head())
    
    