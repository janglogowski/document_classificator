import os
import json
import yaml
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from sklearn.metrics import classification_report, accuracy_score


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


# ----------------- Main train function ----------------- #
def main():
    import argparse

    ap = argparse.ArgumentParser(description="CNN doc-type classifier (resnet18) — legacy-compatible for identical accuracy.")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--level", type=str, default="level2", choices=["level1", "level2", "level3"], help="Scan difficulty level")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-split", action="store_true", default=True,
                    help="Reuse saved split indices to guarantee identical accuracy across runs (default: True)")
    args = ap.parse_args()

    # Set random seed for reproducibility
    set_global_seed(args.seed)

    # --- Load config / paths --- #
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ROOT = os.path.abspath(cfg["paths"]["project_root"])

    data_dir = os.path.join(ROOT, cfg["paths"]["data"]["scans"]["docs"][args.level])
    out_dir  = os.path.join(ROOT, cfg["paths"]["data"]["models"]["cnn_doc_type_classifier"])
    os.makedirs(out_dir, exist_ok=True)

    model_path   = os.path.join(out_dir, f"cnn_doc_type_classifier_{args.level}.pth")
    metrics_path = os.path.join(out_dir, f"cnn_doc_type_classifier_metrics_{args.level}.json")
    split_file   = os.path.join(out_dir, f"split_{args.level}.json")

    print(f"DATA DIR : {data_dir}")
    print(f"OUT DIR  : {out_dir}")
    print(f"SPLIT    : {split_file}")

    # --- Transforms --- #
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # --- Dataset --- #
    base = datasets.ImageFolder(data_dir, transform=train_tf)
    class_names = base.classes
    num_classes = len(class_names)

    n_total = len(base)
    n_val   = int(args.val_split * n_total)
    n_train = n_total - n_val
    if n_val <= 0 or n_train <= 0:
        raise ValueError(f"val_split={args.val_split} results in too small dataset: n_train={n_train}, n_val={n_val}")

    if args.reuse_split and os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        train_idx = saved["train_idx"]
        val_idx   = saved["val_idx"]
        if len(train_idx) + len(val_idx) != n_total:
            raise RuntimeError("Split size mismatch vs current dataset. Remove split_*.json and rerun to save a new split.")
    else:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]
        if args.reuse_split:
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump({"train_idx": train_idx, "val_idx": val_idx}, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved split indices >>>> {split_file}")

    train_set = Subset(base, train_idx)
    val_set   = Subset(base, val_idx)

    base_train = datasets.ImageFolder(data_dir, transform=train_tf)
    base_val   = datasets.ImageFolder(data_dir, transform=val_tf)
    train_set = Subset(base_train, train_idx)
    val_set   = Subset(base_val,   val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Model (ResNet18)  --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # --- Optimizer and loss function --- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    # --- Training loop --- #
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == targets).sum().item()

        train_loss = tr_loss / len(train_set)
        train_acc  = tr_correct / len(train_set)

        # Validation phase
        model.eval()
        val_correct, val_loss_sum = 0, 0.0
        v_preds, v_true = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                loss = criterion(out, targets)
                val_loss_sum += loss.item() * imgs.size(0)
                preds = out.argmax(1)
                val_correct += (preds == targets).sum().item()
                v_preds.extend(preds.cpu().tolist())
                v_true.extend(targets.cpu().tolist())

        val_loss = val_loss_sum / len(val_set)
        val_acc  = val_correct / len(val_set)

        print(f"Epoch {epoch}/{args.epochs} | tr_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Saved best model >>>> {model_path}")

    # Final evaluation
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            preds = out.argmax(1)
            val_preds.extend(preds.cpu().tolist())
            val_true.extend(targets.tolist())

    acc = accuracy_score(val_true, val_preds)
    rep = classification_report(val_true, val_preds, target_names=class_names, output_dict=True)

    print("\n=== FINAL VALIDATION REPORT ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(val_true, val_preds, target_names=class_names))

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "level": args.level,
        "num_classes": num_classes,
        "classes": class_names,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "final_val_accuracy": acc,
        "classification_report": rep,
        "history": history,
        "seed": args.seed,
        "val_split": args.val_split,
        "reuse_split": args.reuse_split,
    }
    with open(metrics_path, "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics >>>> {metrics_path}")
    print(f"Best model    >>>> {model_path}")


if __name__ == "__main__":
    main()
