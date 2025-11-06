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
import SimpleITK as sitk
from typing import List, Dict, Any
from utils import create_save_nifti

def load_dicom_volume_from_list(file_list: List[str]) -> Dict[str, Any]:
    """
    Carrega uma série DICOM já listada (mesma série) e retorna o volume 3D.
    Retorna: img (SimpleITK Image), arr (np.ndarray [Z,Y,X]), spacing (X,Y,Z),
             origin, direction, meta, files (ordenados).
    """
    if not file_list:
        raise ValueError("Lista de DICOMs vazia.")
    files_sorted = _sort_dicom_files(file_list)

    # (opcional) validar se pertencem à mesma série
    first = pydicom.dcmread(files_sorted[0], stop_before_pixels=True, force=True)
    uid = getattr(first, "SeriesInstanceUID", None)
    for p in files_sorted[1:]:
        ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        if uid and getattr(ds, "SeriesInstanceUID", None) != uid:
            raise ValueError("Arquivos pertencem a séries diferentes.")

    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames(files_sorted)
    img3d = reader.Execute()

    return {
        "img": img3d,
        "arr": sitk.GetArrayFromImage(img3d),   # (Z,Y,X)
        "files": files_sorted,
        "spacing": img3d.GetSpacing(),          # (X,Y,Z)
        "origin": img3d.GetOrigin(),
        "direction": img3d.GetDirection(),
        "meta": {
            "SeriesDescription": getattr(first, "SeriesDescription", ""),
            "ProtocolName": getattr(first, "ProtocolName", ""),
            "SeriesNumber": getattr(first, "SeriesNumber", ""),
            "SeriesInstanceUID": uid,
        },
    }

def _sort_dicom_files(file_list: List[str]) -> List[str]:
    """Ordena por ImagePositionPatient (Z) ou InstanceNumber."""
    def key(p):
        ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        inum = getattr(ds, "InstanceNumber", None)
        return int(inum) if inum is not None else os.path.basename(p)
    return sorted(file_list, key=key)

if __name__ == '__main__':
    root_path = 'data/ExamesArya'
    root_output = 'data/ExamesArya_NIFTI2'
    # root_path = 'data/EXAMES/Exames_DICOM'
    # output_path = 'data/EXAMES/Exames_Separados/11517/11517'
    
    patients = os.listdir(root_path)
    patients_error = []
    for patient in tqdm(patients):
        patient = '105655'
        print(patient)
        patient_path = os.path.join(root_path, patient)
        output_path = os.path.join(root_output, patient)
        os.makedirs(output_path, exist_ok=True)
        # Load dicoms
        files = os.listdir(patient_path)
        non_gated_files = [f for f in files if 'SG' in f and f.endswith('.dcm')]
        label_files = [f for f in files if 'IA' in f and f.endswith('.dcm')]
        gated_files = [p for p in files if p not in non_gated_files and p not in label_files]
        
        print(non_gated_files)
        print('-' * 30)
        print(label_files)
        print('-' * 30)
        print(gated_files)
        
        gated_vol = load_dicom_volume_from_list([os.path.join(patient_path, f) for f in gated_files])
        print(gated_vol["arr"].shape, gated_vol["spacing"])
        gated_np = gated_vol["arr"]
        print(np.min(gated_np), np.max(gated_np), np.mean(gated_np))
        label_vol = load_dicom_volume_from_list([os.path.join(patient_path, f) for f in label_files])
        non_gated_vol = load_dicom_volume_from_list([os.path.join(patient_path, f) for f in non_gated_files])
        non_gated_arr = non_gated_vol["arr"]
        print(non_gated_arr.min(), non_gated_arr.max(), non_gated_arr.mean())
        print("Saving NIFTI files...")
        sitk.WriteImage(gated_vol["img"], os.path.join(output_path, f"{patient}_gated.nii.gz"))
        sitk.WriteImage(label_vol["img"], os.path.join(output_path, f"{patient}_mask.nii.gz"))
        sitk.WriteImage(non_gated_vol["img"], os.path.join(output_path, f"{patient}_non_gated.nii.gz"))
    print('finished')
    
    print('Errors found in patients:')
    print(patients_error)
