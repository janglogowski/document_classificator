import os
import json
import yaml
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

#--- load config ---#
CFG  = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT = os.path.abspath(CFG["paths"]["project_root"])

#--- user params ---#
level              = "level3"
samples_per_sub    = 43
batch_size         = 16
num_epochs         = 3
lr                 = 1e-4
seed               = 42
num_workers        = 0

#--- paths ---#
DOCS_DIR    = os.path.join(ROOT, CFG["paths"]["data"]["scans"]["docs"][level])
DRW_DIR     = os.path.join(ROOT, CFG["paths"]["data"]["scans"]["technical_drawings"])
OUT_DIR     = os.path.join(ROOT, CFG["paths"]["data"]["models"]["cnn_doc_vs_drw"])
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(OUT_DIR, f"cnn_doc_vs_drw_{level}.pth")
METRICS_PATH = os.path.join(OUT_DIR, f"cnn_doc_vs_drw_metrics_{level}.json")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

random.seed(seed)
torch.manual_seed(seed)

#--- collect files ---#
def collect_doc_images(doc_root, n_per_subfolder):
    paths = []
    for sub in sorted(os.listdir(doc_root)):
        sub_path = os.path.join(doc_root, sub)
        if not os.path.isdir(sub_path):
            continue
        imgs = [os.path.join(sub_path, f) for f in os.listdir(sub_path)
                if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue
        random.shuffle(imgs)
        paths.extend(imgs[:n_per_subfolder])
    return paths

def collect_all_images(root_dir):
    return [os.path.join(dp, f) for dp, _, fs in os.walk(root_dir) for f in fs if f.lower().endswith(IMG_EXTS)]

doc_imgs = collect_doc_images(DOCS_DIR, samples_per_sub)
drw_imgs = collect_all_images(DRW_DIR)

print(f"Docs picked: {len(doc_imgs)}  (expected ~{samples_per_sub*6})")
print(f"Drawings:    {len(drw_imgs)}")

all_paths  = doc_imgs + drw_imgs
all_labels = [0]*len(doc_imgs) + [1]*len(drw_imgs)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
train_idx, val_idx = next(sss.split(all_paths, all_labels))
train_paths  = [all_paths[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]
val_paths    = [all_paths[i] for i in val_idx]
val_labels   = [all_labels[i] for i in val_idx]

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(5, translate=(0.02,0.02)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#--- dataset ---#
class ImgDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

train_ds = ImgDataset(train_paths, train_labels, train_tf)
val_ds   = ImgDataset(val_paths,   val_labels,   val_tf)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

#--- MobileNetV2 model ---#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0.0
history = []

#--- training loop ---#
for epoch in range(1, num_epochs+1):
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

    model.eval()
    val_loss, val_correct = 0.0, 0
    v_preds, v_true = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            out = model(imgs)
            loss = criterion(out, targets)
            val_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            val_correct += (preds == targets).sum().item()
            v_preds.extend(preds.cpu().tolist())
            v_true.extend(targets.cpu().tolist())

    val_loss /= len(val_ds)
    val_acc = val_correct / len(val_ds)
    print(f"Epoch {epoch}/{num_epochs} | tr_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
    history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[INFO] Saved best model → {MODEL_PATH}")

#--- evaluation ---#
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
    "level": level,
    "n_train": len(train_ds),
    "n_val":   len(val_ds),
    "final_val_accuracy": acc,
    "classification_report": report,
    "history": history,
    "samples_per_subfolder": samples_per_sub}

with open(METRICS_PATH, "w", encoding="utf-8") as jf:
    json.dump(metrics, jf, indent=2, ensure_ascii=False)

print(f"\nSaved metrics >>>> {METRICS_PATH}")
print(f"Best model >>>> {MODEL_PATH}")
