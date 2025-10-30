import matplotlib.pyplot as plt
import os
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import json
from utils import get_basename, load_nifti_sitk, create_save_nifti
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def extract_ids_mask(mask, ids):
    mask_tmp = np.zeros_like(mask)
    min_id = min(ids)
    for id in ids:
        mask_tmp[mask == id] = min_id
    return mask_tmp

def process_patient(args_tuple):
    """
    Processa um único paciente. Função isolada para permitir multiprocessing.
    
    Parameters:
    -----------
    args_tuple : tuple
        (patient, root_path, output_path, totalseg_classes, rois, exclude_files, keywords)
    """
    patient, root_path, output_path, totalseg_classes, rois, exclude_files, keywords = args_tuple
    
    try:
        print(f"Worker processing: {patient}")
        patient_path = os.path.join(root_path, patient)
        nifti_files = os.listdir(patient_path)
        motion_filename = get_basename(nifti_files, exclude_files=exclude_files, keywords=keywords)
        print(f"Processing {motion_filename} - Patient {patient}")

        # Needs to use nibabel for total segmentator
        input_img = nib.load(os.path.join(patient_path, motion_filename))

        output_img = totalsegmentator(
            input_img,
            task='total',
            roi_subset=rois,
            # nr_threads_resampling=1,  # força single-thread
            # nr_threads_saving=1,
            force_split=False,         # desabilita split multiprocess
            quiet=True,
            skip_saving=True,
            device='gpu')
        output_array = output_img.get_fdata()
        
        # Extract masks
        cardio_ids = [51]
        ribs_ids = list(range(92, 116))
        vertebra_ids = list(range(26, 50))
        esternum_ids = [116, 117]
        
        heart_array = extract_ids_mask(output_array, cardio_ids)
        output_array[output_array == 51] = 0
        ribs_array = extract_ids_mask(output_array, ribs_ids)
        vertebra_array = extract_ids_mask(output_array, vertebra_ids)
        esternum_array = extract_ids_mask(output_array, esternum_ids)

        # Dilate masks (cv2.dilate é 2D, aplicar por fatia)
        dilation_kernel = np.ones((3,3), np.uint8)
        heart_dilation_kernel = np.ones((10,10), np.uint8)
        
        # def dilate_3d(arr, kernel, iters):
        #     out = np.zeros_like(arr)
        #     for z in range(arr.shape[2]):
        #         out[:,:,z] = cv2.dilate(arr[:,:,z].astype(np.uint8), kernel, iterations=iters)
        #     return out
        
        heart_array = cv2.dilate(heart_array, dilation_kernel, 5)
        ribs_array = cv2.dilate(ribs_array, dilation_kernel, 3)
        vertebra_array = cv2.dilate(vertebra_array, dilation_kernel, 2)
        esternum_array = cv2.dilate(esternum_array, dilation_kernel, 2)

        bones_array = ribs_array + vertebra_array + esternum_array
        heart_big_array = cv2.dilate(heart_array, heart_dilation_kernel, 3)
        
        # Save outputs
        patient_output_path = os.path.join(output_path, patient)
        os.makedirs(patient_output_path, exist_ok=True)
        create_save_nifti(heart_array, output_img.affine, os.path.join(patient_output_path, 'non_gated_HeartSegs.nii.gz'))
        create_save_nifti(bones_array, output_img.affine, os.path.join(patient_output_path, 'non_gated_BonesSegs.nii.gz'))
        create_save_nifti(heart_big_array, output_img.affine, os.path.join(patient_output_path, 'non_gated_HeartSegs_dilat_k=10.nii.gz'))

        print(f'✓ Saved {patient}: HeartSegs, BonesSegs, HeartSegs_dilat_k=10')
        return (patient, True, None)
    
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"✗ Failed {patient}: {error_msg}")
        traceback.print_exc()
        return (patient, False, error_msg)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Segment Heart and Bones from NIfTI files using multiprocessing')
    parser.add_argument('--root_path', type=str, default='data/ExamesArya_NIFTI2', help='Root path to the NIfTI files')
    parser.add_argument('--output_path', type=str, default='data/ExamesArya_NIFTI2', help='Output path for the segmented images')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    args = parser.parse_args()
    
    # # --- INÍCIO DA MODIFICAÇÃO ---
    # # Força o download dos modelos no processo principal antes de iniciar os workers.
    # print("="*60)
    # print("Verificando e baixando modelos do TotalSegmentator (se necessário)...")
    # try:
    #     # Cria uma imagem NIfTI falsa para acionar o download
    #     dummy_img = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int16), np.eye(4))
    #     # Executa uma tarefa leve para garantir que os modelos sejam baixados
    #     totalsegmentator(dummy_img, output=None, task='total', quiet=False, skip_saving=True, device='gpu')
    #     print("Modelos do TotalSegmentator prontos.")
    # except Exception as e:
    #     print(f"Falha ao pré-carregar os modelos: {e}")
    #     print("O script continuará, mas pode haver downloads repetidos em cada processo.")
    # print("="*60)
    
    # Determinar número de workers
    num_workers = args.num_workers if args.num_workers else multiprocessing.cpu_count()
    print(f"Using {num_workers} worker(s)")

    patients = os.listdir(args.root_path)
    print(f"Total patients to process: {len(patients)}")

    # Load json file with the TotalSeg classes
    json_path = 'TotalSeg_classes.json'
    with open(json_path, 'r') as file:
        totalseg_classes = json.load(file)
    
    # Prepare ROIs
    cardio_ids = [51]
    ribs_ids = list(range(92, 116))
    vertebra_ids = list(range(26, 50))
    esternum_ids = [116, 117]
    
    cardio_classes = [totalseg_classes[str(id)] for id in cardio_ids]
    ribs_classes = [totalseg_classes[str(id)] for id in ribs_ids]
    vertebra_classes = [totalseg_classes[str(id)] for id in vertebra_ids]
    esternum_classes = [totalseg_classes[str(id)] for id in esternum_ids]
    rois = cardio_classes + ribs_classes + vertebra_classes + esternum_classes
    
    # --- INÍCIO DA MODIFICAÇÃO ---
    # Força o download dos modelos no processo principal antes de iniciar os workers.
    print("="*60)
    print("Verificando e baixando modelos do TotalSegmentator (se necessário)...")
    try:
        # Cria uma imagem NIfTI falsa para acionar o download
        dummy_img = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int16), np.eye(4))
        # Executa uma tarefa leve para garantir que os modelos sejam baixados
        totalsegmentator(dummy_img, output=None, task='total', roi_subset=rois, quiet=False, skip_saving=True, device='gpu')
        print("Modelos do TotalSegmentator prontos.")
    except Exception as e:
        print(f"Falha ao pré-carregar os modelos: {e}")
        print("O script continuará, mas pode haver downloads repetidos em cada processo.")
    print("="*60)
    
    exclude_files = ['_HeartSegs', '_FakeGated', '_FakeGated_CircleMask', 'multi_label', 'multi_lesion', 'binary_lesion', '_mask']
    keywords = ['non_gated']

    # Preparar argumentos para cada paciente
    patient_args = [
        (patient, args.root_path, args.output_path, totalseg_classes, rois, exclude_files, keywords)
        for patient in patients
    ]

    # Processar em paralelo com ProcessPoolExecutor (não usa daemon processes)
    results = []
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(process_patient, arg): arg[0] for arg in patient_args}
        
        with tqdm(total=len(patients), desc="Processing patients") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    # Resumo dos resultados
    successful = [r[0] for r in results if r[1]]
    failed = [(r[0], r[2]) for r in results if not r[1]]

    print("\n" + "="*60)
    print(f"Segmentation finished!")
    print(f"Successful: {len(successful)}/{len(patients)}")
    print(f"Failed: {len(failed)}/{len(patients)}")
    
    if failed:
        print("\nFailed patients:")
        for patient, error in failed:
            print(f"  - {patient}: {error}")