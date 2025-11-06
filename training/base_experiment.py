from __future__ import annotations
import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, Optional, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.utils import combine_lesion_region_preds
from tqdm import tqdm

@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"  # 'min' (smaller is better) or 'max'
    monitor: str = "val_loss"  # metric key to watch

    def is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)


class EarlyStopper:
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.best: float = math.inf if config.mode == "min" else -math.inf
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, current: float) -> bool:
        if self.config.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
            return True  # improved
        self.num_bad_epochs += 1
        print(f"Early Stopping: {self.num_bad_epochs}/{self.config.patience}")
        if self.num_bad_epochs >= self.config.patience:
            self.should_stop = True
        return False  # not improved


class BaseExperiment:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        early_stopping: Optional[EarlyStoppingConfig] = None,
        experiment_dir: str = "data/exp1"
    ):
        self.model = model.mtal.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.metrics = metrics or {}
        self.early_stopping = EarlyStopper(
            early_stopping if early_stopping else EarlyStoppingConfig()
        )
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "weights"), exist_ok=True)
        self.best_checkpoint_path = os.path.join(self.experiment_dir, "weights", "best.pt")
        self.last_checkpoint_path = os.path.join(self.experiment_dir, "weights", "last.pt")
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        metric_sums = {k: 0.0 for k in self.metrics}
        count = 0
        total_loss_sum = 0.0
        multi_les_loss_sum = 0.0
        binary_les_loss_sum = 0.0
        for batch in tqdm(dataloader, desc="Training"):
            inputs, multi_les_targets, binary_les_targets = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            y_region, y_lesion = self.model(inputs)
            multi_les_loss = self.criterion(y_region, multi_les_targets)
            binary_les_loss = self.criterion(y_lesion, binary_les_targets)
            batch_size = inputs.size(0)
            
            loss = multi_les_loss + binary_les_loss
            loss.backward()
            self.optimizer.step()
            total_loss_sum += loss.item() * batch_size
            multi_les_loss_sum += multi_les_loss.item() * batch_size
            binary_les_loss_sum += binary_les_loss.item() * batch_size
            

            multi_les_pred = torch.softmax(y_region, dim=1)
            binary_les_pred = torch.softmax(y_lesion, dim=1)
            print(binary_les_pred.shape, multi_les_pred.shape)
            y_pred = multi_les_pred * binary_les_pred[:, 0].unsqueeze(1)
            for name, fn in self.metrics.items():
                with torch.no_grad():
                    metric_sums[name] += fn(y_pred.detach(), multi_les_targets) * batch_size
            count += batch_size
        avg = {
            "train_total_loss": total_loss_sum / max(count, 1),
            "train_multi_les_loss": multi_les_loss_sum / max(count, 1),
            "train_binary_les_loss": binary_les_loss_sum / max(count, 1)
        }
        for name, v in metric_sums.items():
            avg[f"train_{name}"] = v / max(count, 1)
        return avg

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metric_sums = {k: 0.0 for k in self.metrics}
        count = 0
        total_loss_sum = 0.0
        multi_les_loss_sum = 0.0
        binary_les_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                inputs, multi_les_targets, binary_les_targets = self._unpack_batch(batch)
                y_region, y_lesion = self.model(inputs)
                multi_les_loss = self.criterion(y_region, multi_les_targets)
                binary_les_loss = self.criterion(y_lesion, binary_les_targets)
                batch_size = inputs.size(0)
                
                
                loss = multi_les_loss + binary_les_loss
                total_loss_sum += loss.item() * batch_size
                multi_les_loss_sum += multi_les_loss.item() * batch_size
                binary_les_loss_sum += binary_les_loss.item() * batch_size

                multi_les_pred = torch.softmax(y_region, dim=1)
                binary_les_pred = torch.softmax(y_lesion, dim=1)
                y_pred = multi_les_pred * binary_les_pred[:, 1:].unsqueeze(1)
                for name, fn in self.metrics.items():
                    metric_sums[name] += fn(y_pred.detach(), multi_les_targets) * batch_size
                count += batch_size
        avg = {
            "val_total_loss": total_loss_sum / max(count, 1),
            "val_multi_les_loss": multi_les_loss_sum / max(count, 1),
            "val_binary_les_loss": binary_les_loss_sum / max(count, 1)}
        for k, v in metric_sums.items():
            avg[f"val_{k}"] = v / max(count, 1)
        return avg

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
        start_epoch = 1
        if resume and os.path.isfile(self.last_checkpoint_path):
            ckpt = self.load_checkpoint(self.last_checkpoint_path)
            start_epoch = ckpt["epoch"] + 1
            self.early_stopping.best = ckpt.get("best_metric", self.early_stopping.best)
        try:
            for epoch in range(start_epoch, epochs + 1):
                t0 = time.time()
                train_stats = self.train_epoch(train_loader)
                val_stats = self.validate_epoch(val_loader)
                if self.scheduler:
                    # Some schedulers depend on val loss, adjust if necessary
                    try:
                        self.scheduler.step(val_stats["val_loss"])
                    except TypeError:
                        self.scheduler.step()
                epoch_stats = {
                    "epoch": epoch,
                    **train_stats,
                    **val_stats,
                    "lr": self._current_lr(),
                    "time_sec": time.time() - t0,
                }
                self.history.append(epoch_stats)
                improved = self.early_stopping.step(
                    epoch_stats[self.early_stopping.config.monitor]
                )
                # self.save_checkpoint(
                #     self.last_checkpoint_path,
                #     epoch=epoch,
                #     best_metric=self.early_stopping.best,
                #     is_best=False,
                # )
                if improved:
                    #TODO: Verificar pq o modelo não está sendo salvo
                    self.save_checkpoint(
                        self.best_checkpoint_path,
                        epoch=epoch,
                        best_metric=self.early_stopping.best,
                        is_best=True,
                    )
                print(
                    f"{'(improved)' if improved else ''}[{epoch}/{epochs}] lr={epoch_stats['lr']:.6f} | "
                    f"train_loss={epoch_stats['train_total_loss']:.4f} | train_multi_les_loss={epoch_stats['train_multi_les_loss']:.4f}\n | train_binary_les_loss={epoch_stats['train_binary_les_loss']:.4f}\n | "
                    f"val_loss={epoch_stats['val_total_loss']:.4f} | val_multi_les_loss={epoch_stats['val_multi_les_loss']:.4f}\n | val_binary_les_loss={epoch_stats['val_binary_les_loss']:.4f}\n"
                    f"train_acc={epoch_stats.get('train_accuracy', float('nan')):.4f} | val_acc={epoch_stats.get('val_accuracy', float('nan')):.4f}\n"
                    f"train_f1_score={epoch_stats.get('train_f1_score', float('nan')):.4f} | val_f1_score={epoch_stats.get('val_f1_score', float('nan')):.4f} | "
                    f"val_prec={epoch_stats.get('val_precision', float('nan')):.4f} | "
                    f"val_recall={epoch_stats.get('val_recall', float('nan')):.4f} | "
                    f"train_mIoU={epoch_stats.get('train_mIoU', float('nan')):.4f} | val_mIoU={epoch_stats.get('val_mIoU', float('nan')):.4f}\n"
                )
                if self.early_stopping.should_stop:
                    print("Early stopping triggered.")
                    break
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        self._save_history_json()
        return self.history

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        best_metric: float,
        is_best: bool,
    ):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": best_metric,
            "is_best": is_best,
            "early_stopping": {
                "best": self.early_stopping.best,
                "num_bad_epochs": self.early_stopping.num_bad_epochs,
            },
        }
        torch.save(state, path)
        print("model saved to", path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        es = ckpt.get("early_stopping", {})
        if es:
            self.early_stopping.best = es.get("best", self.early_stopping.best)
            self.early_stopping.num_bad_epochs = es.get(
                "num_bad_epochs", self.early_stopping.num_bad_epochs
            )
        return ckpt

    def _current_lr(self) -> float:
        for group in self.optimizer.param_groups:
            return group.get("lr", float("nan"))
        return float("nan")

    def _save_history_json(self):
        hist_path = os.path.join(self.experiment_dir, "history.json")
        # print(self.history)
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _unpack_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Expect (inputs, targets) or dict with keys
        if isinstance(batch, dict):
            inputs = batch["image"]
            multi_les_targets = batch["multi_lesions"]
            binary_les_targets = batch["binary_lesions"]
        else:
            inputs, multi_les_targets, binary_les_targets = batch
        return inputs.to(self.device, non_blocking=True), multi_les_targets.to(
            self.device, non_blocking=True
        ), binary_les_targets.to(self.device, non_blocking=True)

    def load_best(self):
        if os.path.isfile(self.best_checkpoint_path):
            self.load_checkpoint(self.best_checkpoint_path)
        else:
            print("Best checkpoint not found.")

# Example usage (pseudo):
# experiment = BaseExperiment(model, optimizer, criterion, metrics={'mae': lambda yhat,y: torch.nn.functional.l1_loss(yhat, y)})
# history = experiment.fit(train_loader, val_loader, epochs=100)
