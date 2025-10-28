import numpy as np
import nibabel as nib
import os
import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import SimpleITK as sitk
from typing import List, Dict, Any
import re

# def load_nifti_image(file_path: str) -> np.ndarray:
#     """Carrega uma imagem NIfTI e retorna o array numpy."""
#     nifti_img = nib.load(file_path)
#     return nifti_img.get_fdata()

def load_nifti_sitk(file_path: str, return_numpy: bool = True):
    """
    Carrega uma imagem NIfTI usando SimpleITK e opcionalmente a converte para numpy array.
    
    Parameters:
    -----------
    file_path : str
        Caminho para o arquivo NIfTI (.nii ou .nii.gz)
    return_numpy : bool, optional (default=True)
        Se True, também retorna o array numpy além do objeto SimpleITK.Image
        
    Returns:
    --------
    sitk_image : SimpleITK.Image
        Objeto de imagem do SimpleITK
    np_array : numpy.ndarray, optional
        Array numpy com os dados (retornado apenas se return_numpy=True)
        Formato do array: (Z, Y, X)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    sitk_image = sitk.ReadImage(file_path)
    
    if return_numpy:
        # GetArrayFromImage retorna em ordem (Z,Y,X)
        np_array = sitk.GetArrayFromImage(sitk_image)
        return sitk_image, np_array.transpose(1, 2, 0)  # Transpose para (X,Y,Z)
    else:
        return sitk_image
    
def get_basename(files, exclude_files, keywords):
    # Exclude files based on the exclude_files list
    files = [file for file in files if not any(f in file for f in exclude_files)]

    if gated_exam_basename := [
        file for file in files if any(keyword in file for keyword in keywords)]:
        return gated_exam_basename[0]
    else:
        raise ValueError("No matching files found.")
    
def get_partes_moles_basename(files):
    exclude_files=['partes_moles_HeartSegs', 'partes_moles_FakeGated_CircleMask', 'multi_label', 'multi_lesion', 'binary_lesion']
    files = [file for file in files if not any(f in file for f in exclude_files)]
    gated_exam_basename = [file for file in files if 'partes_moles_body' in file or 'mediastino' in file]
    return gated_exam_basename[0]

def set_string_parameters(avg):
    if avg > 0:
        if avg == 4:
            partes_moles_basename = 'partes_moles_FakeGated_avg_slices=4'
            avg_str = 'avg=4'
        elif avg == 3:
            partes_moles_basename = 'partes_moles_FakeGated_mean_slice=3mm'
            avg_str = 'avg=3'
    else:
        partes_moles_basename = 'partes_moles_FakeGated'
        avg_str = 'All Slices'
    return partes_moles_basename, avg_str

def calculate_area(mask, axis=2):
    mask_tmp = mask.copy()
    mask_tmp[mask_tmp != 0] = 1
    return mask_tmp.sum(axis=axis)

def create_save_nifti(data, affine, output_path):
    new_nifti = nib.Nifti1Image(data, affine)
    nib.save(new_nifti, output_path)
    
def save_slice_as_dicom(
    slice_array: np.ndarray, 
    output_folder: str, 
    filename: str,
    patient_id: str = "Unknown",
    slice_position: int = 1,
    series_description: str = "Generated Slice",
    window_center: int = 40,
    window_width: int = 350
):
    """
    Salva um slice 2D como arquivo DICOM.
    
    Parameters:
    -----------
    slice_array : np.ndarray
        Array 2D (H, W) com o slice a ser salvo
    output_folder : str
        Pasta onde salvar o arquivo DICOM
    filename : str
        Nome do arquivo (ex: "slice_001.dcm")
    patient_id : str
        ID do paciente
    slice_position : int
        Posição do slice na série
    series_description : str
        Descrição da série
    window_center : int
        Centro da janela para visualização
    window_width : int
        Largura da janela para visualização
    """
    
    # Criar diretório se não existir
    os.makedirs(output_folder, exist_ok=True)
    
    # Normalizar array para uint16
    if slice_array.dtype != np.uint16:
        # Normalizar para range 0-4095 (12-bit)
        slice_norm = ((slice_array - slice_array.min()) / 
                     (slice_array.max() - slice_array.min()) * 4095).astype(np.uint16)
    else:
        slice_norm = slice_array
    
    # Criar dataset DICOM
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian
    
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128) # type: ignore
    
    # Patient Information
    ds.PatientName = f"Patient_{patient_id}"
    ds.PatientID = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    
    # Study Information
    now = datetime.datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDescription = "Generated Study"
    
    # Series Information
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.SeriesTime = now.strftime("%H%M%S")
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = series_description
    ds.SeriesNumber = "1"
    
    # Instance Information
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()
    ds.InstanceNumber = str(slice_position)
    ds.SliceLocation = str(slice_position)
    
    # Image Information
    ds.Modality = "CT"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = slice_norm.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    
    # Window/Level Information
    ds.WindowCenter = str(window_center)
    ds.WindowWidth = str(window_width)
    
    # Pixel Spacing (default 1mm x 1mm)
    ds.PixelSpacing = ["1.0", "1.0"]
    ds.SliceThickness = "1.0"
    
    # Add pixel data
    ds.PixelData = slice_norm.tobytes()
    
    # Save file
    output_path = os.path.join(output_folder, filename)
    ds.save_as(output_path)
    print(f"Saved DICOM: {output_path}")

# def _sort_dicom_files(file_list: List[str]) -> List[str]:
#     """Ordena por ImagePositionPatient (Z) ou InstanceNumber."""
#     def key(p):
#         ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
#         ipp = getattr(ds, "ImagePositionPatient", None)
#         if ipp is not None and len(ipp) == 3:
#             return float(ipp[2])
#         inum = getattr(ds, "InstanceNumber", None)
#         return int(inum) if inum is not None else os.path.basename(p)
#     return sorted(file_list, key=key)

def _sort_dicom_files(file_list: List[str]) -> List[str]:
    """Ordena os arquivos pelo número no nome (última sequência de dígitos)."""
    def num_key(p: str) -> int:
        m = re.search(r'(\d+)(?=\D*$)', os.path.basename(p))
        return int(m.group(1)) if m else 10**12  # sem número vai para o fim
    return sorted(file_list, key=num_key)

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