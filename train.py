import torch
import numpy as np
from torch import nn, optim
import os
from MTAL_CACS.src.model import MTALModel
import argparse
from training.dataset import CardiacNIFTIDataset
from torch.utils.data import DataLoader, random_split
from training.base_experiment import BaseExperiment, EarlyStoppingConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de treinamento para o modelo MTAL.")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Nome do experimento.")
    parser.add_argument("--data_dir", type=str, default="data/ExamesArya_CalcSegTraining", help="Diretório dos dados de entrada.")
    parser.add_argument("--output_dir", type=str, default="data/output", help="Diretório para salvar os resultados.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Número de épocas para o treinamento.")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote para o treinamento.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proporção dos dados para validação.")
    parser.add_argument("--decoder_type", type=str, default="both", choices=["both", "binary", "coronaries"], help="Tipo de decoder a ser utilizado.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Taxa de aprendizado para o otimizador.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = CardiacNIFTIDataset(
        root=args.data_dir,
        label_suffix="_mask",
        normalize=True,
        strict_pairs=True)
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = MTALModel(device=device)
    model.create()
    model.load("MTAL_CACS/model/model.pt")

    # Initialize optimizer
    optimizer = optim.Adam(model.mtal.parameters(), lr=args.learning_rate)

    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()  # modelo deve fornecer log-probs, ou faça log(outputs)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    metrics = {
        "accuracy": lambda outputs, labels: (outputs.argmax(dim=1) == labels).float().mean().item(),
        "loss": lambda outputs, labels: criterion(outputs, labels).item(),
        "f1_score": lambda outputs, labels: ((2 * (outputs.argmax(dim=1) * labels).sum()) / ((outputs.argmax(dim=1) + labels).sum() + 1e-6)).item(),
        "precision": lambda outputs, labels: ((outputs.argmax(dim=1) * labels).sum() / (outputs.argmax(dim=1).sum() + 1e-6)).item(),
        "recall": lambda outputs, labels: ((outputs.argmax(dim=1) * labels).sum() / (labels.sum() + 1e-6)).item(),
        "mIoU": lambda outputs, labels: ((outputs.argmax(dim=1) & labels).sum() / (outputs.argmax(dim=1) | labels).sum() + 1e-6).item(),
    }
    
        # Configuração do Early Stopping
    early_cfg = EarlyStoppingConfig(
        patience=10,      # para parar após 10 épocas sem melhora
        min_delta=0.001, # melhora mínima
        mode="min",      # queremos minimizar val_loss
        monitor="val_loss"
    )
    
    # Initialize experiment
    experiment = BaseExperiment(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        metrics=metrics,
        early_stopping=early_cfg,
        checkpoint_dir=ckpt_dir
    )

    # Start training
    experiment.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.num_epochs,
        log_every=1
    )