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
    # Init predictions
    # pred_lesion = np.zeros(image.shape)
    # pred_lesion_multi = np.zeros(image.shape)
    # pred_region = np.zeros(image.shape)
        
    # Combine lesion predictions
    print(Y_lesion.shape)
    Y_lesion_bin = Y_lesion.round()
    print("DEBUG")
    print(Y_lesion[:, 1].shape, Xmask.shape)
    # Filtra predições de lesão pelos candidatos
    Y_lesion_bin[:,1,:,:] = Y_lesion_bin[:,1,:,:]*Xmask
    Y_lesion_bin[:,0,:,:] = 1-Y_lesion_bin[:,1,:,:]
    
    # Combine region predictions
    Y_region_bin = torch.zeros(Y_region.shape, dtype=torch.float32, device=Y_region.device)
    Y_region_bin = Y_region_bin.scatter_(1, torch.argmax(Y_region, dim=1, keepdim=True) , 1)
    # Y_region_multi = torch.argmax(Y_region_bin, dim=1, keepdim=True)

    # Combine lesion and region predictions
    Y_lesion_multi = torch.argmax(torch.cat((torch.max(Y_lesion_bin[:,0:1,:,:], Y_region_bin[:,0:1,:,:]), Y_lesion_bin[:,1:2,:,:].repeat((1,3,1,1)) * Y_region_bin[:,1:,:,:]), dim=1), dim=1, keepdim=True)
    
    # Fill predictions
    # pred_lesion = Y_lesion_bin[0,1,:,:]
    # pred_region = Y_region_multi[0,0,:,:]
    print("DEBUG 2")
    print(Y_lesion_multi.shape)
    # pred_lesion_multi = Y_lesion_multi[:, 0, :, :]
    print(torch.unique(Y_lesion_multi))
    
    return Y_lesion_multi[:, 0, :, :].to(torch.float32)