import argparse
import os
import json
import math
import yaml
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ----------------- utils ----------------- #

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _models_root_for(cfg: dict, key: str) -> Path:
    base_rel = cfg["paths"]["data"]["models"][key]
    return Path(cfg["paths"]["project_root"]) / base_rel


def _tcrit_95(df: int) -> float:
    table = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365,
             8:2.306, 9:2.262, 10:2.228, 15:2.131, 20:2.086, 30:2.042}
    if df in table: return table[df]
    if df < 1: return 12.706
    return 1.96


def _ci95(scores: List[float]) -> Tuple[float, float]:
    k = len(scores)
    if k == 0:
        return (0.0, 0.0)
    mean = float(np.mean(scores))
    if k == 1:
        return (mean, mean)
    std_sample = float(np.std(scores, ddof=1))
    se = std_sample / math.sqrt(k)
    tcrit = _tcrit_95(k - 1)
    lo = max(0.0, mean - tcrit * se)
    hi = min(1.0, mean + tcrit * se)
    return (lo, hi)


def _build_transforms(img_size: int = 256):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def _compute_class_weight_from_indices(indices: np.ndarray, targets: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency weights (normalized) na bazie etykiet z 'indices'.
    """
    counts = np.bincount(targets[indices], minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # guard
    weights = (counts.sum() / counts) / num_classes  # suma wag ~1
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_model(num_classes: int, lr: float, class_weight: torch.Tensor | None = None):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # Fine-tuning tylko głowy (szybko i stabilnie)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Optymalizujemy tylko warstwę fc
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # CE z wagami klas (jeśli są)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    return model, optimizer, criterion


def _train_one(model, optimizer, criterion, device, train_loader, val_loader, epochs: int):
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tr_loss, tr_correct, n_train = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == targets).sum().item()
            n_train += imgs.size(0)
        train_loss = tr_loss / max(1, n_train)
        train_acc  = tr_correct / max(1, n_train)

        # val
        model.eval()
        val_correct, val_loss_sum, n_val = 0, 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                out = model(imgs)
                loss = criterion(out, targets)
                val_loss_sum += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == targets).sum().item()
                n_val += imgs.size(0)
        val_loss = val_loss_sum / max(1, n_val)
        val_acc  = val_correct / max(1, n_val)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, history


# ----------------- main ----------------- #

def main():
    ap = argparse.ArgumentParser(description="CNN doc-type classifier (ResNet18) with unified CV metrics.")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--level", type=str, choices=["level1", "level2", "level3"], default="level2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--cv-splits", type=int, default=None, help="Override cv_splits from config (set 0/1 to disable CV).")
    ap.add_argument("--val-split", type=float, default=None, help="Holdout split ratio if CV disabled.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-split", action="store_true", default=True,
                    help="Reuse/save indices for deterministic splits.")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (Windows: 0 najbezpieczniej).")
    args = ap.parse_args()

    set_global_seed(args.seed)

    # config / paths
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))
    data_dir = ROOT / cfg["paths"]["data"]["scans"]["docs"][args.level]
    out_dir  = _models_root_for(cfg, "cnn_doc_type_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path   = out_dir / f"cnn_doc_type_{args.level}.pth"
    metrics_path = out_dir / f"cnn_doc_type_metrics_{args.level}.json"
    names_path   = out_dir / "class_names.txt"

    # transforms & dataset
    train_tf, val_tf = _build_transforms(args.img_size)
    base_for_targets = datasets.ImageFolder(str(data_dir), transform=val_tf)
    class_names = base_for_targets.classes
    num_classes = len(class_names)

    with open(names_path, "w", encoding="utf-8") as f:
        for n in class_names:
            f.write(n + "\n")

    targets = np.array(base_for_targets.targets)
    n_total = len(base_for_targets)
    if n_total == 0:
        raise RuntimeError(f"No images found under {data_dir}")

    # CV/holdout params
    cnn_cfg = cfg.get("cnn_doc_type_classifier", {})
    cv_k = args.cv_splits if args.cv_splits is not None else int(cnn_cfg.get("cv_splits", 5))
    holdout_enabled = (cv_k is None) or (cv_k <= 1)
    val_split = args.val_split if args.val_split is not None else float(cnn_cfg.get("val_split", 0.2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = bool(torch.cuda.is_available())

    # storage for metrics
    cv_fold_accs: List[float] = []
    oof_pred = np.full(shape=(n_total,), fill_value=-1, dtype=np.int64)

    split_file = out_dir / f"cnn_doc_type_splits_{args.level}.json"

    # --------------- CV --------------- #
    if not holdout_enabled:
        if args.reuse_split and split_file.exists():
            saved = json.loads(split_file.read_text(encoding="utf-8"))
            folds = saved.get("folds", [])
            if len(folds) != cv_k:
                raise RuntimeError("Saved CV folds count mismatch. Remove the split JSON to regenerate.")
        else:
            skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=args.seed)
            folds = []
            for tr_idx, va_idx in skf.split(np.zeros(n_total), targets):
                folds.append({"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist()})
            if args.reuse_split:
                split_file.write_text(json.dumps({"folds": folds}, ensure_ascii=False, indent=2), encoding="utf-8")

        # run folds
        for fi, fold in enumerate(folds, 1):
            tr_idx = np.array(fold["train_idx"], dtype=np.int64)
            va_idx = np.array(fold["val_idx"], dtype=np.int64)

            # Data
            train_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
            val_ds   = datasets.ImageFolder(str(data_dir), transform=val_tf)
            train_set = Subset(train_ds, tr_idx.tolist())
            val_set   = Subset(val_ds, va_idx .tolist())

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=pin)
            val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=pin)

            # class weights per fold
            cw = _compute_class_weight_from_indices(tr_idx, targets, num_classes, device)

            model, optimizer, criterion = _make_model(num_classes, args.lr, class_weight=cw)
            model.to(device)

            best_val_acc, _ = _train_one(model, optimizer, criterion, device, train_loader, val_loader, args.epochs)
            cv_fold_accs.append(float(best_val_acc))

            # infer on this fold's val for OOF predictions
            model.eval()
            preds_fold = []
            with torch.no_grad():
                for imgs, _targets in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    out = model(imgs)
                    preds = out.argmax(1).cpu().numpy().tolist()
                    preds_fold.extend(preds)
            oof_pred[va_idx] = np.array(preds_fold, dtype=np.int64)

            print(f"[CV] Fold {fi}/{cv_k} best val acc = {best_val_acc:.4f}")

        assert (oof_pred >= 0).all(), "OOF predictions missing for some samples"

        # final CV metrics
        cv_mean = float(np.mean(cv_fold_accs))
        cv_std  = float(np.std(cv_fold_accs, ddof=0))
        ci_lo, ci_hi = _ci95(cv_fold_accs)

        cv_report = classification_report(targets, oof_pred, target_names=class_names, output_dict=True)

        print("\n=== CV SUMMARY ===")
        print(f"fold acc: {cv_fold_accs}")
        print(f"mean: {cv_mean:.4f} ± {cv_std:.4f}   | 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

        # after CV: train final model on ALL data (save)
        full_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
        full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=pin)
        # class weights on all data
        all_idx = np.arange(n_total, dtype=np.int64)
        cw_full = _compute_class_weight_from_indices(all_idx, targets, num_classes, device)
        model, optimizer, criterion = _make_model(num_classes, args.lr, class_weight=cw_full)
        model.to(device)

        val_full_ds = datasets.ImageFolder(str(data_dir), transform=val_tf)
        val_full_loader = DataLoader(val_full_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=pin)
        _train_one(model, optimizer, criterion, device, full_loader, val_full_loader, args.epochs)
        torch.save(model.state_dict(), model_path)

        final_section = {
            "accuracy": cv_mean,
            "accuracy_ci95": [ci_lo, ci_hi],
            "origin": {"type": "cv_mean", "k": cv_k}
        }
        cv_section = {
            "k": cv_k,
            "random_state": args.seed,
            "accuracy_scores": [float(x) for x in cv_fold_accs],
            "accuracy_mean": cv_mean,
            "accuracy_std": cv_std,
            "classification_report": cv_report
        }
        holdout_section = None

    # --------------- Holdout (no CV) --------------- #
    else:
        idx_all = np.arange(n_total, dtype=np.int64)
        tr_idx, va_idx = train_test_split(
            idx_all, test_size=val_split, stratify=targets, random_state=args.seed
        )

        if args.reuse_split:
            split_file.write_text(json.dumps(
                {"holdout": {"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist()}},
                ensure_ascii=False, indent=2), encoding="utf-8")

        train_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
        val_ds   = datasets.ImageFolder(str(data_dir), transform=val_tf)
        train_set = Subset(train_ds, tr_idx.tolist())
        val_set   = Subset(val_ds, va_idx.tolist())

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin)

        # class weights on train split
        cw = _compute_class_weight_from_indices(tr_idx, targets, num_classes, device)
        model, optimizer, criterion = _make_model(num_classes, args.lr, class_weight=cw)
        model.to(device)

        best_val_acc, history = _train_one(model, optimizer, criterion, device, train_loader, val_loader, args.epochs)

        # final eval on val set
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, targets_b in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                out  = model(imgs)
                preds = out.argmax(1).cpu().tolist()
                val_preds += preds
                val_true  += targets_b.tolist()

        acc = accuracy_score(val_true, val_preds)
        rep = classification_report(val_true, val_preds, target_names=class_names, output_dict=True)

        torch.save(model.state_dict(), model_path)

        final_section = {
            "accuracy": float(acc),
            "origin": {"type": "holdout", "ratio": float(val_split)}
        }
        cv_section = None
        holdout_section = {
            "test_size": float(val_split),
            "train_size": int(len(train_set)),
            "test_size_abs": int(len(val_set)),
            "accuracy": float(acc),
            "classification_report": rep
        }

    # --------------- Save metrics --------------- #
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "level": args.level,
        "num_classes": num_classes,
        "classes": class_names,
        "img_size": args.img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "reuse_split": bool(args.reuse_split),
        "final": final_section,
        "cv": cv_section,
        "holdout": holdout_section,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics  >>>> {metrics_path}")
    print(f"Saved model    >>>> {model_path}")
    print(f"Saved classes  >>>> {names_path}")


if __name__ == "__main__":
    main()
