import os
import yaml
import json
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

#--- load config ---#
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT = os.path.abspath(cfg["paths"]["project_root"])

level  = "level3"               # level1/level2/level3
engine = "tesseract_ocr"        # "tesseract_ocr" or "easy_ocr"

#--- text input folders ---#
DOCS_TXT_FOLDER = os.path.join(ROOT,cfg["paths"]["data"]["extracted"][engine][level])
DRW_TXT_FOLDER = os.path.join(ROOT,cfg["paths"]["data"]["extracted"][engine]["technical_drawings"])

OUT_MODEL_DIR = os.path.join(ROOT,cfg["paths"]["data"]["models"]["lr_doc_vs_drw"],engine)
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(OUT_MODEL_DIR, f"lr_doc_vs_drw_{level}.pkl")
VECT_PATH    = os.path.join(OUT_MODEL_DIR, f"doc_vs_drw_vect_{level}.pkl")
METRICS_PATH = os.path.join(OUT_MODEL_DIR, f"doc_vs_drw_metrics_{level}.json")

print(f"Document folder: {DOCS_TXT_FOLDER}")
print(f"Drawing  folder: {DRW_TXT_FOLDER}")
print(f"Saving model to:   {MODEL_PATH}")

#--- collect random docs per subfolder ---#
docs_paths = []
for sub in os.listdir(DOCS_TXT_FOLDER):
    subp = os.path.join(DOCS_TXT_FOLDER, sub)
    if not os.path.isdir(subp):
        continue
    txts = [
        os.path.join(subp, f)
        for f in os.listdir(subp)
        if f.lower().endswith(".txt")]
    docs_paths += random.sample(txts, min(43, len(txts)))

#--- collect drawings ---#
drw_txts  = [
    os.path.join(DRW_TXT_FOLDER, f)
    for f in os.listdir(DRW_TXT_FOLDER)
    if f.lower().endswith(".txt")]
drw_paths = random.sample(drw_txts, min(258, len(drw_txts)))

print(f"Collected {len(docs_paths)} document samples and {len(drw_paths)} drawing samples")

#--- read texts & labels ---#
texts, labels = [], []
for p in docs_paths:
    t = open(p, encoding="utf-8").read().strip()
    if t:
        texts.append(t)
        labels.append("document")
for p in drw_paths:
    t = open(p, encoding="utf-8").read().strip()
    if t:
        texts.append(t)
        labels.append("tech_drw")

print("Label distribution:", dict(Counter(labels)))

#--- TF–IDF ---#
tv_cfg = cfg["doc_vs_drw_classifier"]["tfidf"]
vectorizer = TfidfVectorizer(
    stop_words=tv_cfg.get("stop_words"),
    max_features=tv_cfg["max_features"],
    ngram_range=(1, 2),)

X = vectorizer.fit_transform(texts)
y = labels

print(f"TF–IDF matrix shape: {X.shape}")

#--- train/test split ---#
split_cfg = cfg["doc_vs_drw_classifier"]["train_test_split"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=split_cfg["test_size"],
    stratify=y if split_cfg.get("stratify") else None,
    random_state=split_cfg["random_state"])

print(f"Train/test sizes: {X_train.shape[0]}/{X_test.shape[0]}")

#--- train logistic regression ---#
lr_cfg = cfg["doc_vs_drw_classifier"]["logistic_regression"]
model = LogisticRegression(**lr_cfg)
model.fit(X_train, y_train)

#--- evaluation ---#
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print("\n=== EVALUATION ===")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

#--- save artifacts ---#
joblib.dump(model,      MODEL_PATH)
joblib.dump(vectorizer, VECT_PATH)
print(f"\nSaved model      → {MODEL_PATH}")
print(f"Saved vectorizer → {VECT_PATH}")

#--- save metrics ---#
metrics = {
    "engine": engine,
    "level": level,
    "n_docs": len(docs_paths),
    "n_drawings": len(drw_paths),
    "train_size": X_train.shape[0],
    "test_size":  X_test.shape[0],
    "accuracy":   acc,
    "classification_report": report,
    "tfidf_max_features":    tv_cfg["max_features"],
    "logistic_regression_params": lr_cfg}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"Saved metrics >>>> {METRICS_PATH}")