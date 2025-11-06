import torch
import numpy as np
from torch import nn, optim
import os
from MTAL_CACS.src.model import MTALModel
import argparse
from training.dataset import CardiacNIFTIDataset
from torch.utils.data import DataLoader, random_split
from training.base_experiment import BaseExperiment, EarlyStoppingConfig
from training.utils import accuracy, precision_macro, recall_macro, f1_macro, miou
import pandas as pd
from training.focal_loss import FocalLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de treinamento para o modelo MTAL.")
    parser.add_argument("--exp_name", type=str, default="data/exp1", help="Nome do experimento.")
    parser.add_argument("--data_dir", type=str, default="data/ExamesArya_CalcSegTraining", help="Diretório dos dados de entrada.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Número de épocas para o treinamento.")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Tamanho do lote para o treinamento.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proporção dos dados para validação.")
    parser.add_argument("--decoder_type", type=str, default="both", choices=["both", "binary", "coronaries"], help="Tipo de decoder a ser utilizado.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Taxa de aprendizado para o otimizador.")
    parser.add_argument("--map_labels", action="store_true", help="Mapeia os labels do dataset para os labels esperados pelo modelo.")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Caminho para o arquivo CSV de treino.")
    parser.add_argument("--num_workers", type=int, default=4, help="Número de workers para DataLoader.")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"], help="Tipo de scheduler de taxa de aprendizado.")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"], help="Tipo de função de perda.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Valor de gamma para Focal Loss (se usada).")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Valor de alpha para Focal Loss (se usada).")
    args = parser.parse_args()

    exp_dir = os.path.join("data/experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # ckpt_dir = os.path.join(args.exp_name, "checkpoints")
    # os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    df_train = pd.read_csv(args.train_csv)
    dataset = CardiacNIFTIDataset(
        root=args.data_dir,
        label_suffix="_mask",
        df_sample=df_train,
        normalize=True,
        strict_pairs=True,
        map_labels=args.map_labels
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split train / val
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Training set size: {len(train_set)} samples")
    print(f"Validation set size: {len(val_set)} samples")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Initialize model
    model = MTALModel(device=device)
    model.create()
    model.load("MTAL_CACS/model/model.pt")

    # Initialize optimizer
    optimizer = optim.Adam(model.mtal.parameters(), lr=args.learning_rate)

    # Initialize loss function
    if args.loss == "ce":
        multi_les_criterion = nn.CrossEntropyLoss()
        # bin_les_criterion = nn.CrossEntropyLoss()
        bin_les_criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "focal":
        class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
        class_weights_bin = torch.tensor([1-args.focal_alpha, args.focal_alpha]).to(device)
        multi_les_criterion = FocalLoss(mode="multiclass", gamma=args.focal_gamma, alpha=class_weights)
        bin_les_criterion = FocalLoss(mode="multiclass", gamma=args.focal_gamma, alpha=class_weights_bin)
        # bin_les_criterion = FocalLoss(mode="binary", gamma=args.focal_gamma, alpha=class_weights_bin)

    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    metrics = {
        "accuracy": accuracy,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1_score": f1_macro,
        "mIoU": miou,
    }
    
        # Configuração do Early Stopping
    early_cfg = EarlyStoppingConfig(
        patience=10,      # para parar após 10 épocas sem melhora
        min_delta=0.001, # melhora mínima
        mode="min",      # queremos minimizar val_total_loss
        monitor="val_total_loss"
    )
    
    # Initialize experiment
    experiment = BaseExperiment(
        model=model,
        optimizer=optimizer,
        multi_les_criterion=multi_les_criterion,
        bin_les_criterion=bin_les_criterion,
        device=device,
        scheduler=scheduler,
        metrics=metrics,
        early_stopping=early_cfg,
        experiment_dir=exp_dir
    )

    # Start training
    experiment.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.num_epochs
    )