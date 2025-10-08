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
    # Y_lesion = torch.max(Y_lesion, dim=1, keepdim=True).values[:, 0, :, :]
    Y_lesion = torch.argmax(Y_lesion, dim=1, keepdim=True)

    Y_lesion = Y_lesion * Xmask
    
    Y_lesion_multi = Y_region.clone()
    Y_lesion_multi[:, 0, :, :] = Y_region[:, 0, :, :] * Y_lesion
    Y_lesion_multi[:, 1, :, :] = Y_region[:, 1, :, :] * Y_lesion
    Y_lesion_multi[:, 2, :, :] = Y_region[:, 2, :, :] * Y_lesion
    Y_lesion_multi[:, 3, :, :] = Y_region[:, 3, :, :] * Y_lesion
    
    return Y_lesion_multi