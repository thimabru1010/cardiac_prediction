import os
import SimpleITK as sitk
from pathlib import Path
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# adiciona a pasta pai ao PYTHONPATH em runtime
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import load_dicom_volume_from_list  # agora funciona
# ...existing code...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select exams with 0 CAC score from DICOM series.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing DICOM series.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder for selected exams.")
    parser.add_argument("--scores_csv", type=str, default="data/cac_score_data_0score.csv", help="Path to CSV file with CAC scores.")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    already_processed = os.listdir(output_folder)

    # List all subfolders in the input folder
    patient_ids = os.listdir(input_folder)
    print(f"Total patients found: {len(patient_ids)}")
    df = pd.read_csv(args.scores_csv)
    _0score_ids = df['ID'].astype(str).tolist()
    patient_ids = [p for p in patient_ids if p in _0score_ids]
    print(f"Patients with 0 score in CSV: {len(_0score_ids)}")
    
    patient_ids = [p for p in patient_ids if p not in already_processed]
    print(f"Already processed patients: {len(already_processed)}")
    print(f"Patients to process: {len(patient_ids)}")
    
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        print(f"Patient {patient_id}")
        patient_path = os.path.join(input_folder, patient_id)
        dicom_files = os.listdir(patient_path)
        if os.path.isdir(os.path.join(patient_path, dicom_files[0])):
            print(f"Patient with 2 subfolders: {patient_id}")
            patient_path = os.path.join(patient_path, patient_id)
            dicom_files = os.listdir(patient_path)
        print(patient_path)

        # Collect all DICOM files in the patient folder
        # dicom_files = [os.path.join(patient_path, f) for f in os.listdir(patient_path) if f.lower().endswith('.dcm')]
        # dicom_files = os.listdir(patient_path)
        
        # Separate files in Gated (G) and Non-Gated (SG)
        sg_prefix = 'SG'
        non_gated_files = sorted([f for f in dicom_files if 'SG' in f])
        if len(non_gated_files) == 0:
            sg_prefix = 'NG'
            non_gated_files = sorted([f for f in dicom_files if 'NG' in f])
        gated_files = sorted([f for f in dicom_files if 'G' in f and 'L' not in f and sg_prefix not in f])
        if len(gated_files) == 0:
            gated_files = sorted([f for f in dicom_files if 'D' in f and 'L' not in f and sg_prefix not in f])
        
        # print(non_gated_files)
        # print(gated_files)

        # Load DICOM volume
        gated_vol = load_dicom_volume_from_list([os.path.join(patient_path, f) for f in gated_files])
        non_gated_vol = load_dicom_volume_from_list([os.path.join(patient_path, f) for f in non_gated_files])

        # Save as Niftis
        output_patient_folder = os.path.join(output_folder, patient_id)
        os.makedirs(output_patient_folder, exist_ok=True)
        sitk.WriteImage(gated_vol["img"], os.path.join(output_patient_folder, f"{patient_id}_gated.nii.gz"))
        sitk.WriteImage(non_gated_vol["img"], os.path.join(output_patient_folder, f"{patient_id}_non_gated.nii.gz"))
