import numpy as np
import pandas as pd
import argparse
import os

def read_files(data_dir):
    patients = os.listdir(data_dir)
    filenames = {"patient_id": [], "slice_name": []}
    for p in patients:
        slices_filenames = os.listdir(os.path.join(data_dir, p))
        slices_filenames = sorted(slices_filenames)
        exam_filename = [f for f in slices_filenames if '_ct' in f]
        label_filename = [f for f in slices_filenames if '_mask' in f]
        for ct_filename, mask_filename in zip(exam_filename, label_filename):
            ct_slice_id = ct_filename.split('_ct.npy')[0]
            mask_slice_id = mask_filename.split('_mask.npy')[0]
            if ct_slice_id != mask_slice_id:
                print(f"Warning: CT slice ID {ct_slice_id} does not match mask slice ID {mask_slice_id}. Skipping.")
                continue
            filenames["patient_id"].append(os.path.join(p))
            filenames["slice_name"].append(ct_slice_id)
    return filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para dividir dataset em treino e teste.")
    parser.add_argument("--data_dir", type=str, default="data/ExamesArya_CalcSegTraining", help="Diretório dos dados de entrada.")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodução da divisão.")
    args = parser.parse_args()

    filenames = read_files(args.data_dir)
    df = pd.DataFrame(filenames)

    # Dividir em treino e teste
    train_data = df.sample(frac=0.7, random_state=args.seed)
    test_data = df.drop(train_data.index)
    
    print(train_data.head())
    print("-"*50)
    print(test_data.head())
    print("-"*50)
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    

    # Salvar os dados divididos
    train_data.to_csv(os.path.join("data", "train.csv"), index=False)
    test_data.to_csv(os.path.join("data", "test.csv"), index=False)
