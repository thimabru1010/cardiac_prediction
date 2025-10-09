from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

def map_labels_to_original(mask: np.ndarray) -> np.ndarray:
    """
    Mapeia os labels do modelo (0,1,2,3) para os labels originais (0,1,2,3)
    onde:
        0 - fundo
        1 - LAD
        2 - LCX
        3 - RCA
    """
    label_mapping = {
        0: 0,  # fundo
        1: 2,  # LAD
        2: 1,  # LCX
        3: 3   # RCA
    }
    mapped_mask = np.copy(mask)
    for src_label, tgt_label in label_mapping.items():
        mapped_mask[mask == src_label] = tgt_label
    return mapped_mask
class CardiacNIFTIDataset(Dataset):
    """
    Dataset básico para ler NIFTI de exames cardíacos.
    Estrutura esperada:
        data/ExamesArya_NIFTI_CalcSegTraining/
            paciente01.nii.gz
            paciente01_seg.nii.gz   (opcional, se existir segmentação)
            paciente02.nii.gz
            paciente02_seg.nii.gz
            ...

    Args:
        root (str | Path): Caminho da pasta base (ex: 'data/ExamesArya_NIFTI_CalcSegTraining').
        label_suffix (str): Sufixo que identifica o arquivo de segmentação (default: '_seg').
        normalize (bool): Se True normaliza cada volume (z-score).
        transform (Callable): Transform adicional aplicada ao dict retornado.
        load_affine (bool): Retorna affine no dict.
        strict_pairs (bool): Se True descarta casos sem segmentação.
    Retorno __getitem__:
        {
            'image': Tensor (C,D,H,W) ou (C,H,W) dependendo dimensionalidade,
            'label': Tensor ou None,
            'id': str (nome base),
            'affine': Tensor (opcional)
        }
    """
    def __init__(
        self,
        root: str | Path = "data/ExamesArya_CalcSegTraining",
        label_suffix: str = "_mask",
        normalize: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        load_affine: bool = False,
        strict_pairs: bool = False,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {self.root}")
        self.label_suffix = label_suffix
        self.normalize = normalize
        self.transform = transform
        self.load_affine = load_affine
        self.strict_pairs = strict_pairs

        self.samples: List[Tuple[Path, Optional[Path], str]] = []
        self._index_files()

    def _index_files(self):
        patients = os.listdir(self.root)
        for p in patients:
            slices_filenames = os.listdir(os.path.join(self.root, p))
            slices_filenames = sorted(slices_filenames)
            exam_filename = [f for f in slices_filenames if '_ct' in f]
            label_filename = [f for f in slices_filenames if '_mask' in f]
            for ct_filename, mask_filename in zip(exam_filename, label_filename):
                ct_slice_id = ct_filename.split('_ct.npy')[0]
                mask_slice_id = mask_filename.split('_mask.npy')[0]
                if ct_slice_id != mask_slice_id:
                    print(f"Warning: CT slice ID {ct_slice_id} does not match mask slice ID {mask_slice_id}. Skipping.")
                    continue
                self.samples.append((Path(os.path.join(self.root, p, ct_filename)), Path(os.path.join(self.root, p, mask_filename)), ct_slice_id))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npy(self, path: Path) -> np.ndarray:
        array = np.load(path)
        return array

    def _normalize(self, t: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return t
        # Normalize image data
        Xmin = -2000
        Xmax = 1300
        t[t==-3024]=-2048
        t = (t - Xmin) / (Xmax - Xmin)
        return t
        
    # def _normalize(self, t: torch.Tensor) -> torch.Tensor:
    #     if not self.normalize:
    #         return t
    #     # Normalização por volume (ignora zeros)
    #     mean = t.mean()
    #     std = t.std()
    #     if std > 0:
    #         t = (t - mean) / std
    #     return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, label_path, sample_id = self.samples[idx]
        image_tensor = self._load_npy(image_path)
        image_tensor_norm = self._normalize(image_tensor)

        label_tensor = None
        if label_path is not None:
            label_tensor = self._load_npy(label_path)
            label_tensor = map_labels_to_original(label_tensor)
            # exclude Other Calcifications for while (class 4)
            label_tensor[label_tensor == 4] = 0
            # Converte para long (segmentações)
            label_tensor = torch.from_numpy(label_tensor).squeeze(0).long()  # remove canal se for 1
        
        # print("Image tensor shape before unsqueeze:", image_tensor.shape)
        # print("Label tensor shape after unsqueeze:", label_tensor.shape )
        image_tensor = torch.from_numpy(image_tensor)
        image_tensor_norm = torch.from_numpy(image_tensor_norm)
        calc_candidates = torch.zeros_like(image_tensor)
        calc_candidates[image_tensor >= 130] = 1
        input_image = torch.stack((image_tensor_norm.to(torch.float32), calc_candidates.to(torch.float32)), dim=0).to(torch.float32)  # add channel dim if missing
        sample = {
            "image": input_image,
            "label": label_tensor,
            "id": sample_id,
        }
        
        # print(image_tensor.dtype, label_tensor.dtype)
        # print(image_tensor.shape)
        if self.transform:
            sample = self.transform(sample)
        return sample


# Exemplo de uso (remover ou adaptar conforme necessidade):
# if __name__ == "__main__":
#     ds = CardiacNIFTIDataset(root="data/ExamesArya_NIFTI_CalcSegTraining")
#     item = ds[0]
#     print(item["id"], item["image"].shape, item["label"].shape if item["label"] is not None else None)
