import torch
import numpy as np
import os
from MTAL_CACS.src.model import MTALModel
import argparse
import pandas as pd
from utils import load_nifti_sitk
import SimpleITK as sitk
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de inferência para o modelo MTAL.")
    parser.add_argument("--model_path", type=str, required=True, help="Caminho para o modelo treinado.")
    parser.add_argument("--data_dir", type=str, required=True, help="Diretório dos dados de entrada.")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório para salvar as previsões.")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamanho do lote para a inferência.")
    parser.add_argument("--fake_gated", action="store_true", help="Se deve inferir exames com gated falso.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load best model
    model = MTALModel(device=device)
    model.create()
    model.load_checkpoint(args.model_path)
    model = model.mtal.to(device)
    model.eval()
    
    # Loop through data and make predictions
    error_patients = []
    patients = os.listdir(args.data_dir)
    for patient in tqdm(patients, desc="Processing patients"):
        # print(f'Processing patient: {patient}')
        # Load patient data
        filename = f'{patient}_gated.nii.gz'
        input_exam_path = os.path.join(args.data_dir, patient, filename)
        if args.fake_gated:
            print('Inferring Fake Gated exam for patient')
            filename = 'non_gated_FakeGated_avg_slices=4.nii.gz'
            input_exam_path = os.path.join(args.data_dir, patient, filename)
        
        input_img, input_arr = load_nifti_sitk(input_exam_path, return_numpy=True)
        input_tensor = torch.from_numpy(input_arr).to(torch.float32).to(device)  # Add batch and channel dimensions
        
        # Make predictions
        binary_preds = []
        region_preds = []
        try:
            for bs in range(0, input_tensor.shape[0], args.batch_size):
                input_batch = input_tensor[bs:bs + args.batch_size].unsqueeze(1)  # Add channel dimension
                with torch.no_grad():
                    region_lesions, binary_lesions = model(input_batch)
                    
                    region_lesions = torch.softmax(region_lesions, dim=1)
                    binary_lesions = torch.softmax(binary_lesions, dim=1)
                    
                    region_lesions = torch.argmax(region_lesions, dim=1)
                    binary_lesions = torch.argmax(binary_lesions, dim=1)
                    
                    region_preds.append(region_lesions.detach().cpu().numpy())
                    binary_preds.append(binary_lesions.detach().cpu().numpy())
        except Exception as e:
            print(f"Error processing patient {patient}: {e}")
            error_patients.append(patient)
            continue
        region_preds = np.concatenate(region_preds, axis=0)
        binary_preds = np.concatenate(binary_preds, axis=0)
        multi_lesion_pred = region_preds * binary_preds
        # Save predictions with sitk
        basename = filename.split('.')[0]
        output_region_path = os.path.join(args.output_dir, f'{basename}_region_lesion_trained_model.nii.gz')
        output_binary_path = os.path.join(args.output_dir, f'{basename}_binary_lesion_trained_model.nii.gz')
        output_multi_lesion_path = os.path.join(args.output_dir, f'{basename}_multi_lesion_trained_model.nii.gz')
        
        sitk_region = sitk.GetImageFromArray(region_preds.astype(np.uint8))
        sitk_binary = sitk.GetImageFromArray(binary_preds.astype(np.uint8))
        sitk_multi_lesion = sitk.GetImageFromArray(multi_lesion_pred.astype(np.uint8))
        
        sitk_region.CopyInformation(input_img)
        sitk_binary.CopyInformation(input_img)
        sitk_multi_lesion.CopyInformation(input_img)
        
        sitk.WriteImage(sitk_region, output_region_path)
        sitk.WriteImage(sitk_binary, output_binary_path)
        sitk.WriteImage(sitk_multi_lesion, output_multi_lesion_path)
        # print(f'Saved predictions for patient {patient}')
    print("Processing completed.")
    if error_patients:
        print("Patients with errors during processing:")
        for patient in error_patients:
            print(f"- {patient}")