import os
import json
import yaml
import random
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


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


def save_json(path: str, obj: Dict) -> None:
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=convert)



def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------- Data collection ------------------------- #
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def list_images(root: str) -> List[str]:
    out = []
    for dp, _, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def collect_doc_images_deterministic(doc_root: str, n_per_subfolder: int, seed: int) -> List[str]:

    rng = random.Random(seed)
    selected = []

    for sub in sorted(os.listdir(doc_root)):
        sub_path = os.path.join(doc_root, sub)
        if not os.path.isdir(sub_path):
            continue
        imgs = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue
        imgs = sorted(imgs)
        # Shuffle with a subfolder-dependent seed (stable)
        sub_rng = random.Random((seed, sub).__hash__())
        sub_rng.shuffle(imgs)
        take = imgs[:min(n_per_subfolder, len(imgs))]
        selected.extend(take)

    return selected


def collect_drawings_deterministic(drw_root: str, cap: int, seed: int) -> List[str]:
    all_imgs = list_images(drw_root)
    rng = random.Random(seed)
    if len(all_imgs) <= cap:
        return all_imgs
    idx = list(range(len(all_imgs)))
    rng.shuffle(idx)
    pick = [all_imgs[i] for i in idx[:cap]]
    return sorted(pick)


# ------------------------- Dataset ------------------------- #
class ImgDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ------------------------- Training ------------------------- #
def main():
    import argparse

    ap = argparse.ArgumentParser(description="CNN doc vs technical drawing (MobileNetV2) — reproducible accuracy.")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--level", type=str, default="level2", choices=["level1", "level2", "level3"])
    ap.add_argument("--samples-per-sub", type=int, default=43, help="Docs per subfolder (deterministic)")
    ap.add_argument("--drawings-cap", type=int, default=258, help="Max drawings to sample (deterministic)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--reuse-selection", action="store_true", default=True,
                    help="Reuse saved selection & split for identical accuracy (default: True)")
    args = ap.parse_args()

    set_global_seed(args.seed)

    # --- Load config & paths --- #
    CFG = yaml.safe_load(open(args.config, encoding="utf-8"))
    ROOT = os.path.abspath(CFG["paths"]["project_root"])

    DOCS_DIR = os.path.join(ROOT, CFG["paths"]["data"]["scans"]["docs"][args.level])
    DRW_DIR  = os.path.join(ROOT, CFG["paths"]["data"]["scans"]["technical_drawings"])
    OUT_DIR  = os.path.join(ROOT, CFG["paths"]["data"]["models"]["cnn_doc_vs_drw"])
    os.makedirs(OUT_DIR, exist_ok=True)

    MODEL_PATH    = os.path.join(OUT_DIR, f"cnn_doc_vs_drw_{args.level}.pth")
    METRICS_PATH  = os.path.join(OUT_DIR, f"cnn_doc_vs_drw_metrics_{args.level}.json")
    SELECTION_JSON = os.path.join(OUT_DIR, f"selection_doc_vs_drw_{args.level}.json")
    SPLIT_JSON     = os.path.join(OUT_DIR, f"split_doc_vs_drw_{args.level}.json")

    print(f"Docs dir : {DOCS_DIR}")
    print(f"Drw dir  : {DRW_DIR}")
    print(f"Out dir  : {OUT_DIR}")

    if args.reuse_selection and os.path.exists(SELECTION_JSON):
        sel = load_json(SELECTION_JSON)
        doc_imgs = sel["doc_imgs"]
        drw_imgs = sel["drw_imgs"]
        print("[INFO] Reusing saved selection:", SELECTION_JSON)
    else:
        doc_imgs = collect_doc_images_deterministic(DOCS_DIR, args.samples_per_sub, args.seed)
        drw_imgs = collect_drawings_deterministic(DRW_DIR, args.drawings_cap, args.seed)
        save_json(SELECTION_JSON, {"doc_imgs": doc_imgs, "drw_imgs": drw_imgs})
        print("[INFO] Saved selection:", SELECTION_JSON)

    print(f"Docs selected   : {len(doc_imgs)}  (expected ~{args.samples_per_sub} × #subfolders)")
    print(f"Drawings selected: {len(drw_imgs)}")

    all_paths  = doc_imgs + drw_imgs
    all_labels = [0] * len(doc_imgs) + [1] * len(drw_imgs)  # 0=doc, 1=tech_drw

    if args.reuse_selection and os.path.exists(SPLIT_JSON):
        sp = load_json(SPLIT_JSON)
        train_idx = sp["train_idx"]
        val_idx   = sp["val_idx"]
        print("[INFO] Reusing saved split:", SPLIT_JSON)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, val_idx = next(sss.split(all_paths, all_labels))
        save_json(SPLIT_JSON, {"train_idx": train_idx, "val_idx": val_idx})
        print("[INFO] Saved split:", SPLIT_JSON)

    train_paths  = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths    = [all_paths[i] for i in val_idx]
    val_labels   = [all_labels[i] for i in val_idx]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # --- Transforms --- #
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(5, translate=(0.02, 0.02)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # --- Datasets / loaders --- #
    train_ds = ImgDataset(train_paths, train_labels, train_tf)
    val_ds   = ImgDataset(val_paths,   val_labels,   val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model: MobileNetV2 --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    history = []

    # --- Train loop --- #
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (outputs.argmax(1) == targets).sum().item()

        train_loss = tr_loss / len(train_ds)
        train_acc  = tr_correct / len(train_ds)

        # Validation
        model.eval()
        val_loss_sum, val_correct = 0.0, 0
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

        val_loss = val_loss_sum / len(val_ds)
        val_acc  = val_correct / len(val_ds)

        print(f"Epoch {epoch}/{args.epochs} | tr_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[INFO] Saved best model >>>> {MODEL_PATH}")

    # --- Final eval (on saved best) --- #
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    v_preds, v_true = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            preds = out.argmax(1)
            v_preds.extend(preds.cpu().tolist())
            v_true.extend(targets.tolist())

    acc = accuracy_score(v_true, v_preds)
    report = classification_report(v_true, v_preds, target_names=["doc", "tech_drw"], output_dict=True)

    print("\n=== FINAL VALIDATION REPORT ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(v_true, v_preds, target_names=["doc", "tech_drw"]))

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "level": args.level,
        "n_train": len(train_ds),
        "n_val":   len(val_ds),
        "final_val_accuracy": acc,
        "classification_report": report,
        "history": history,
        "samples_per_subfolder": args.samples_per_sub,
        "drawings_cap": args.drawings_cap,
        "seed": args.seed,
        "reuse_selection": args.reuse_selection,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=2, ensure_ascii=False)

    print(f"\nSaved metrics >>>> {METRICS_PATH}")
    print(f"Best model   >>>> {MODEL_PATH}")


if __name__ == "__main__":
    main()
