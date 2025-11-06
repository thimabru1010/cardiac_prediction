import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

class FocalLoss(nn.Module):
    """
    Focal Loss unificada para:
      - 'binary'   : binário / multi-label (usa sigmóide por logit)
      - 'multiclass': multi-classe exclusiva (usa softmax por logit)

    Parâmetros
    ----------
    mode : str
        'binary' ou 'multiclass'.
    alpha : float | Tensor | None
        Peso(s) de classe. Para 'multiclass', pode ser escalar (mesmo peso
        para todas as classes) ou tensor shape [C] com um alpha por classe.
        Para 'binary', se escalar é interpretado como alpha_pos (alpha_neg = 1-alpha);
        também aceita tensor broadcastável ao shape do alvo.
    gamma : float
        Fator de foco (γ).
    reduction : str
        'none' | 'mean' | 'sum'.
    ignore_index : int | None
        Válido apenas para 'multiclass'. Rótulos iguais a ignore_index são ignorados.
    """
    def __init__(
        self,
        mode: str = "multiclass",
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        if mode not in {"binary", "multiclass"}:
            raise ValueError("mode must be 'binary' or 'multiclass'")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.mode = mode
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.clone().detach())
        else:
            self.alpha = None  # type: ignore

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs:
          - binary     : logits de shape [N, *] (mesmo shape dos targets)
          - multiclass : logits de shape [N, C, ...]
        targets:
          - binary     : {0,1} com shape igual a inputs
          - multiclass : índices inteiros {0..C-1} com shape [N, ...]
        """
        if self.mode == "binary":
            return self._forward_binary(inputs, targets)
        else:
            return self._forward_multiclass(inputs, targets)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _forward_binary(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE com logits (evita instabilidade numérica)
        # ce = BCE(por logit); p = sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(inputs, targets.to(inputs.dtype), reduction="none")
        p = torch.sigmoid(inputs)
        # p_t = p se y=1; 1-p se y=0
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1 - p_t).clamp_min(1e-8).pow(self.gamma)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            if self.alpha.ndim == 0:
                # alpha escalar -> alpha_pos = alpha; alpha_neg = 1 - alpha
                alpha_pos = self.alpha
                alpha_neg = 1.0 - self.alpha
                alpha_t = alpha_pos * targets + alpha_neg * (1 - targets)
            else:
                # alpha tensor broadcastável (ex.: por canal)
                alpha_t = self.alpha

        loss = modulating * ce * alpha_t
        return self._reduce(loss)

    def _forward_multiclass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Multi-classe exclusiva (softmax). Suporta ignore_index.
        inputs: [N, C, *]
        targets: [N, *] com índices em [0, C-1] ou ignore_index
        """
        n, c = inputs.shape[:2]
        # Rearranja para trabalhar por elemento
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        # Máscara de ignore_index (se houver)
        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index)
        else:
            ignore_mask = torch.zeros_like(targets, dtype=torch.bool)

        # Seleciona p_t e log p_t da classe verdadeira
        # targets válidos serão usados para gather
        valid_targets = targets.clone()
        valid_targets[ignore_mask] = 0  # valor dummy para evitar index error no gather

        # p_t/log_p_t com gather
        gather_dims = list(range(2, inputs.ndim))
        idx = (valid_targets.unsqueeze(1)).expand(-1, 1, *inputs.shape[2:])
        p_t = probs.gather(1, idx).squeeze(1)
        log_p_t = log_probs.gather(1, idx).squeeze(1)

        # Modulating factor
        modulating = (1 - p_t).clamp_min(1e-8).pow(self.gamma)

        # Alpha_t
        if self.alpha is None:
            alpha_t = torch.ones_like(p_t)
        else:
            if self.alpha.ndim == 0:
                alpha_t = torch.full_like(p_t, float(self.alpha))
            else:
                # alpha por classe: alpha[targets]
                alpha_vec = self.alpha
                if alpha_vec.numel() != c:
                    raise ValueError(f"alpha must have length C={c} for multiclass.")
                alpha_t = alpha_vec[valid_targets]
        
        # Loss por elemento
        loss = -alpha_t * modulating * log_p_t
        # Zera posições ignoradas
        loss = torch.where(ignore_mask, torch.zeros_like(loss), loss)

        return self._reduce(loss)

#! Exemplo de uso (remover ou adaptar conforme necessidade):
# logits e alvos com o MESMO shape (ex.: [N, 1] ou [N, H, W])
# crit = FocalLoss(mode="binary", alpha=0.25, gamma=3.0, reduction="mean")
# loss = crit(logits, targets.float())

# logits: [N, C, ...]; targets: [N, ...] com índices 0..C-1
# class_weights = torch.tensor([1.0, 2.0, 4.0], device=logits.device)  # alpha por classe (opcional)
# crit = FocalLoss(mode="multiclass", alpha=class_weights, gamma=3.0, reduction="mean", ignore_index=255)
# loss = crit(logits, targets)  # targets pode conter 255 para ignorar (ex.: segmentação)

