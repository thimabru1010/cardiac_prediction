import torch
import numpy as np
from torch import nn, optim
import os
from MTAL_CACS.src.model import MTALModel
import argparse
from training.dataset import CardiacNIFTIDataset
from torch.utils.data import DataLoader, random_split
from training.base_experiment import BaseExperiment, EarlyStoppingConfig
from training.utils import accuracy, precision_macro, recall_macro, f1_macro, miou, combine_lesion_region_preds
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de treinamento para o modelo MTAL.")
    parser.add_argument("--exp_name", type=str, default="data/exp1", help="Nome do experimento.")
    parser.add_argument("--data_dir", type=str, default="data/ExamesArya_CalcSegTraining", help="Diretório dos dados de entrada.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Número de épocas para o treinamento.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote para o treinamento.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proporção dos dados para validação.")
    parser.add_argument("--decoder_type", type=str, default="both", choices=["both", "binary", "coronaries"], help="Tipo de decoder a ser utilizado.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Taxa de aprendizado para o otimizador.")
    parser.add_argument("--pretrained_model", type=str, default="MTAL_CACS/model/model.pt", help="Caminho para o modelo pré-treinado.")
    args = parser.parse_args()

    os.makedirs(args.exp_name, exist_ok=True)
    # ckpt_dir = os.path.join(args.exp_name, "checkpoints")
    # os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    df_test = pd.read_csv(os.path.join("data", "test.csv"))
    test_dataset = CardiacNIFTIDataset(
        root=args.data_dir,
        label_suffix="_mask",
        df_sample=df_test,
        normalize=True,
        strict_pairs=True)
    print(f"Test dataset size: {len(test_dataset)} samples")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load best model
    model = MTALModel(device=device)
    model.create()
    model.load_checkpoint(f"{args.exp_name}/weights/best.pt")
    if args.pretrained_model:
        model.load(args.pretrained_model)

    # Evaluate model
    model = model.mtal.to(device)
    model.eval()
    metrics_sum = {
        'accuracy': 0.0,
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'f1_macro': 0.0,
        'miou': 0.0
    }
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            y_region, y_lesion = model(inputs)
            batch_size = inputs.size(0)
            y_logits = combine_lesion_region_preds(y_lesion, y_region, inputs[:, 1])
            y_pred = torch.softmax(y_logits, dim=1)
            
            metrics_sum['accuracy'] += accuracy(y_pred, labels) * batch_size
            metrics_sum['precision_macro'] += precision_macro(y_pred, labels) * batch_size
            metrics_sum['recall_macro'] += recall_macro(y_pred, labels) * batch_size
            metrics_sum['f1_macro'] += f1_macro(y_pred, labels) * batch_size
            metrics_sum['miou'] += miou(y_pred, labels) * batch_size
            
            count += batch_size
    metrics_avg = {k: v / count for k, v in metrics_sum.items()}
    print("Test set metrics:")
    print(f"Accuracy: {metrics_avg['accuracy']:.4f}")
    print(f"Precision: {metrics_avg['precision_macro']:.4f}")
    print(f"Recall: {metrics_avg['recall_macro']:.4f}")
    print(f"F1 Score: {metrics_avg['f1_macro']:.4f}")
    print(f"mIoU: {metrics_avg['miou']:.4f}")
    
    # Save metrics in a csv
    metrics_df = pd.DataFrame([metrics_avg])
    metrics_df.to_csv(os.path.join(args.exp_name, "test_metrics.csv"), index=False)
    
    
