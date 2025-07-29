import os
import yaml
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# --- load config --- #
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT = os.path.abspath(cfg["paths"]["project_root"])

# --- choose OCR engine and scan level --- #
engine = "tesseract_ocr"
level = "level3"

# --- derive paths --- #
DOCS_FOLDER = os.path.join(ROOT, cfg["paths"]["data"]["extracted"][engine][level])
MODEL_PATH = os.path.join(ROOT, cfg["paths"]["data"]["models"]["lr_doc_type_classifier"][engine], f"doc_type_classifier_{level}.pkl")
VECT_PATH = os.path.join(ROOT, cfg["paths"]["data"]["models"]["lr_doc_type_classifier"][engine], f"doc_type_vect_{level}.pkl")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print(f"Using engine={engine}, level={level}")
print(f"Training data folder: {DOCS_FOLDER}")
print(f"Model will be saved to: {MODEL_PATH}")
print(f"Vectorizer will be saved to: {VECT_PATH}")

# --- gather all .txt and count empties --- #
all_txt = []
empty_txt = []
for root_dir, _, files in os.walk(DOCS_FOLDER):
    for fname in files:
        if not fname.lower().endswith(".txt"):
            continue
        full = os.path.join(root_dir, fname)
        all_txt.append(full)
        text = open(full, encoding="utf-8").read().strip()
        if not text:
            empty_txt.append(full)

print(f"\nTotal .txt files found: {len(all_txt)}")
print(f"Empty and skipped:       {len(empty_txt)}")

# --- collect non-empty docs & labels --- #
docs, labels = [], []
for root_dir, _, files in os.walk(DOCS_FOLDER):
    for fname in files:
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(root_dir, fname)
        text = open(path, encoding="utf-8").read().strip()
        if not text:
            continue
        docs.append(text)
        ln = fname.lower()
        if "quality" in ln:
            labels.append("quality_checklist")
        elif "inspection" in ln:
            labels.append("inspection_report")
        elif "product" in ln:
            labels.append("product_data_sheet")
        elif "bom" in ln:
            labels.append("bill_of_materials")
        elif "maintenance" in ln:
            labels.append("maintenance_log")
        elif "daily" in ln:
            labels.append("daily_report")
        else:
            labels.append("other")

print(f"\nLoaded {len(docs)} documents for training")
dist = Counter(labels)
print("Class distribution (all):", dict(dist))

if not docs:
    raise RuntimeError(f"No .txt files found under {DOCS_FOLDER}")

# --- TF–IDF vectorization --- #
tfidf_cfg = cfg["doc_type_classifier"]["tfidf"]
stop = tfidf_cfg.get("stop_words") or None

vectorizer = TfidfVectorizer(
    stop_words=stop,
    max_features=tfidf_cfg["max_features"],
    ngram_range=(1, 2))

print(f"\nFitting TF–IDF vectorizer (max_features={tfidf_cfg['max_features']})...")
X = vectorizer.fit_transform(docs)
print(f"TF–IDF matrix shape: {X.shape}")

y = labels

# --- train/test split --- #
split_cfg = cfg["doc_type_classifier"]["train_test_split"]
print(f"\nSplitting data (test_size={split_cfg['test_size']})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=split_cfg["test_size"],
    stratify=y if split_cfg.get("stratify") else None,
    random_state=split_cfg["random_state"])

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Class distribution (train):", dict(Counter(y_train)))
print("Class distribution (test):",  dict(Counter(y_test)))

# --- train logistic regression --- #
lr_cfg = cfg["doc_type_classifier"]["logistic_regression"]
print("\nTraining LogisticRegression with config:", lr_cfg)
model = LogisticRegression(**lr_cfg)
model.fit(X_train, y_train)
print("Model training complete")

# --- evaluation --- #
print("\nEvaluation on test set")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# --- cross-validation --- #
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5)  # <<<
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_mean:.4f}, Std: {cv_std:.4f}")

# --- retrain on full data --- #
print("\nRetraining model on full dataset (X, y)...")  # <<<
model.fit(X, y)  # <<<

# --- inspect top features per class --- #
feature_names = vectorizer.get_feature_names_out()
for class_idx, class_label in enumerate(model.classes_):
    coefs = model.coef_[class_idx]
    top10 = coefs.argsort()[-10:][::-1]
    print(f"\nTop predictors for class '{class_label}':")
    for i in top10:
        print(f"  {feature_names[i]}")

# --- save model & vectorizer --- #
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECT_PATH)
print(f"\nSaved model → {MODEL_PATH}")
print(f"Saved vectorizer → {VECT_PATH}")

# --- save metrics to JSON --- #
metrics = {
    "engine": engine,
    "level": level,
    "accuracy": acc,
    "classification_report": report_dict,
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "tfidf_max_features": tfidf_cfg["max_features"],
    "tfidf_stop_words": tfidf_cfg.get("stop_words"),
    "logistic_regression_params": lr_cfg,
    "cross_val_scores": cv_scores.tolist(),
    "cross_val_mean": cv_mean,
    "cross_val_std": cv_std
}

metrics_path = os.path.join(os.path.dirname(MODEL_PATH), f"doc_type_classifier_metrics_{level}.json")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"Saved metrics → {metrics_path}")