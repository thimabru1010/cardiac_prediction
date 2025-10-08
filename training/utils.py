import numpy as np
import torch

def combine_lesion_region_preds(Y_lesion: torch.Tensor, Y_region: torch.Tensor, Xmask: torch.Tensor) -> torch.Tensor:
    """
    Combina as saídas do decoder de lesão e do decoder de região para produzir a predição final.
    
    Args:
        Y_lesion (torch.Tensor): Saída do decoder de lesão com forma (N, 2, H, W).
        Y_region (torch.Tensor): Saída do decoder de região com forma (N, 4, H, W).
        Xmask (torch.Tensor): Máscara binária indicando candidatos a lesão com forma (N, 1, H, W).
        
    Returns:
        torch.Tensor: Predição final com valores 0 (sem lesão) ou 1..3 (regiões de lesão).
    """
    Y_lesion = torch.softmax(Y_lesion, dim=1)
    # print(Y_lesion.shape)
    # Y_lesion = torch.max(Y_lesion, dim=1, keepdim=True).values[:, 0, :, :]
    # Y_lesion = torch.argmax(Y_lesion, dim=1, keepdim=True)
    # print(Y_lesion.shape, Xmask.shape)

    Y_lesion = Y_lesion[:, 1, :, :].unsqueeze(1) * Xmask.unsqueeze(1)
    # print(Y_lesion.shape, Y_region.shape)
    Y_lesion_multi = Y_region.clone()
    Y_lesion_multi = Y_lesion_multi * Y_lesion
    # Y_lesion_multi[:, 0, :, :] = Y_region[:, 0, :, :] * Y_lesion
    # Y_lesion_multi[:, 1, :, :] = Y_region[:, 1, :, :] * Y_lesion
    # Y_lesion_multi[:, 2, :, :] = Y_region[:, 2, :, :] * Y_lesion
    # Y_lesion_multi[:, 3, :, :] = Y_region[:, 3, :, :] * Y_lesion
    
    return Y_lesion_multi

def accuracy(outputs, labels):
    pred = outputs.argmax(dim=1)
    return ((pred == labels).float().mean()).item()

def precision_macro(outputs, labels):
    pred = outputs.argmax(dim=1)
    C = outputs.size(1)
    eps = 1e-6
    vals = []
    for k in range(1, C):  # ignora background=0
        tp = ((pred == k) & (labels == k)).sum().float()
        fp = ((pred == k) & (labels != k)).sum().float()
        # print(f'Debugging: {k}, TP: {tp}, FP: {fp}')
        denom = tp + fp
        if denom > 0:
            prec = tp / (denom + eps)
            vals.append(prec)
    # return torch.stack(vals).mean().item()
    return (torch.stack(vals).mean().item() if len(vals) > 0 else 0.0)

def recall_macro(outputs, labels):
    pred = outputs.argmax(dim=1)
    C = outputs.size(1)
    eps = 1e-6
    vals = []
    for k in range(1, C):
        tp = ((pred == k) & (labels == k)).sum().float()
        fn = ((pred != k) & (labels == k)).sum().float()
        rec = tp / (tp + fn + eps)
        vals.append(rec)
    return torch.stack(vals).mean().item()

def f1_macro(outputs, labels):
    pred = outputs.argmax(dim=1)
    C = outputs.size(1)
    eps = 1e-6
    vals = []
    for k in range(1, C):
        tp = ((pred == k) & (labels == k)).sum().float()
        fp = ((pred == k) & (labels != k)).sum().float()
        fn = ((pred != k) & (labels == k)).sum().float()
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        vals.append(f1)
    return torch.stack(vals).mean().item()

def miou(outputs, labels):
    pred = outputs.argmax(dim=1)
    C = outputs.size(1)
    eps = 1e-6
    ious = []
    for k in range(1, C):
        inter = ((pred == k) & (labels == k)).sum().float()
        union = ((pred == k) | (labels == k)).sum().float()
        ious.append(inter / (union + eps))
    return torch.stack(ious).mean().item()