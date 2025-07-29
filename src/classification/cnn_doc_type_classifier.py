import os
import json
import yaml
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, accuracy_score

#--- load config ---#
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT = os.path.abspath(CFG["paths"]["project_root"])

#--- choose scan level ---#
level = "level3"  # level1 / level2 / level3

#--- directory with images ---#
DATA_DIR = os.path.join(ROOT, CFG["paths"]["data"]["scans"]["docs"][level])

#--- output (model + metrics) ---#
OUT_DIR = os.path.join(ROOT, CFG["paths"]["data"]["models"]["cnn_doc_type_classifier"])
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, f"cnn_doc_type_classifier_{level}.pth")
METRICS_PATH = os.path.join(OUT_DIR, f"cnn_doc_type_classifier_metrics_{level}.json")

#--- hyper params ---#
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42
NUM_WORKERS = 0  

torch.manual_seed(SEED)

#--- transforms ---#
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(3),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std =[0.229, 0.224, 0.225])])

val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std =[0.229, 0.224, 0.225])])

#--- dataset split ---#
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
class_names = full_dataset.classes
num_classes = len(class_names)

n_total = len(full_dataset)
n_val = int(VAL_SPLIT * n_total)
n_train = n_total - n_val

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED))

val_dataset.dataset.transform = val_tf

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)

#--- model (resnet18)---#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
for p in model.parameters():
    p.requires_grad = False
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

#--- training loop ---#
best_val_acc = 0.0
history = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_correct = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()

    train_loss = running_loss / n_train
    train_acc = running_correct / n_train

    #--- validation ---#
    model.eval()
    val_correct = 0
    val_preds = []
    val_true = []
    val_loss_sum = 0.0

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_loss_sum += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            val_correct += (preds == targets).sum().item()

            val_preds.extend(preds.cpu().tolist())
            val_true.extend(targets.cpu().tolist())

    val_loss = val_loss_sum / n_val
    val_acc = val_correct / n_val

    print(f"Epoch {epoch}/{NUM_EPOCHS} "
          f"| train_loss={train_loss:.4f} acc={train_acc:.4f} "
          f"| val_loss={val_loss:.4f} acc={val_acc:.4f}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc})

    #--- save best model ---#
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[INFO] Saved new best model >>>> {MODEL_PATH}")

#--- evaluation ---#
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

val_preds = []
val_true = []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        val_preds.extend(preds.cpu().tolist())
        val_true.extend(targets.tolist())

acc = accuracy_score(val_true, val_preds)
report = classification_report(val_true, val_preds,
                               target_names=class_names,
                               output_dict=True)
print("\n=== FINAL VALIDATION REPORT ===")
print(f"Accuracy: {acc:.4f}")
print(classification_report(val_true, val_preds, target_names=class_names))

#--- save json metrics---#
metrics = {
    "timestamp": datetime.now().isoformat(),
    "level": level,
    "num_classes": num_classes,
    "classes": class_names,
    "n_train": n_train,
    "n_val": n_val,
    "final_val_accuracy": acc,
    "classification_report": report,
    "history": history}

with open(METRICS_PATH, "w", encoding="utf-8") as jf: json.dump(metrics, jf, indent=2, ensure_ascii=False)

print(f"\nSaved metrics >>>> {METRICS_PATH}")
print(f"Best model path >>>> {MODEL_PATH}")
