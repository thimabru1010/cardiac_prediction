from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Any

try:
    import nibabel as nib
except ImportError as e:
    raise ImportError("Você precisa instalar nibabel: pip install nibabel") from e

import torch
from torch.utils.data import Dataset


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
            # TODO: Para cada paciente, loopar em cima dos slices .npy e carregá-los
            exam_filename = os.path.join(self.root, p,  p + "_gated_prep.nii.gz")
            label_filename = os.path.join(self.root, p,  p + f"{self.label_suffix}.nii.gz")
            self.samples.append((Path(exam_filename), Path(label_filename), p))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_nifti(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        img = nib.load(str(path))
        data = img.get_fdata(dtype="float32")  # float32 para PyTorch
        # Adiciona canal se necessário (C,Z,Y,X) ou (C,H,W)
        if data.ndim == 3:
            data_tensor = torch.from_numpy(data).unsqueeze(0)  # (1,D,H,W)
        elif data.ndim == 4:
            # Assume última dimensão como canal ou tempo; reorganiza para (C,D,H,W)
            # Se (X,Y,Z,T) => permutar para (T,Z,Y,X)
            data_tensor = torch.from_numpy(data).permute(3, 2, 1, 0)
        else:
            raise ValueError(f"Dimensionalidade não suportada ({data.ndim}) em {path}")
        # affine_tensor = torch.from_numpy(img.affine).float()
        return data_tensor #, affine_tensor

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
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
        image_tensor = self._load_nifti(image_path)
        image_tensor = self._normalize(image_tensor)

        label_tensor = None
        if label_path is not None:
            lt = self._load_nifti(label_path)
            # Converte para long (segmentações)
            label_tensor = lt.squeeze(0).long()  # remove canal se for 1

        calc_candidates = torch.zeros_like(image_tensor)
        calc_candidates[image_tensor > 130] = 1
        image_tensor = torch.cat((image_tensor, calc_candidates), dim=0)
        sample = {
            "image": image_tensor,
            "label": label_tensor,
            "id": sample_id,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


# Exemplo de uso (remover ou adaptar conforme necessidade):
# if __name__ == "__main__":
#     ds = CardiacNIFTIDataset(root="data/ExamesArya_NIFTI_CalcSegTraining")
#     item = ds[0]
#     print(item["id"], item["image"].shape, item["label"].shape if item["label"] is not None else None)
