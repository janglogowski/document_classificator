import argparse
import os
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


# ------------------------------ IO helpers ------------------------------ #

def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------ Data loading ------------------------------ #

def collect_docs_and_labels(docs_root: Path) -> tuple[list[str], list[str], list[Path]]:
    """
    Walk docs_root and collect (text, label) where label is the first
    path segment under docs_root: docs_root/<LABEL>/file.txt
    """
    texts: list[str] = []
    labels: list[str] = []
    paths: list[Path] = []

    if not docs_root.exists():
        raise FileNotFoundError(f"Docs root does not exist: {docs_root}")

    # Expect subfolders = classes
    for class_dir in sorted([d for d in docs_root.iterdir() if d.is_dir()]):
        label = class_dir.name  # e.g., BOMS, DAILY_REPORTS, ...
        for p in class_dir.rglob("*.txt"):
            if not p.is_file():
                continue
            text = read_text(p).strip()
            if not text:
                continue
            texts.append(text)
            labels.append(label)
            paths.append(p)

    return texts, labels, paths


# ------------------------------ Main training ------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Train LR doc-type classifier from OCR text.")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--engine", type=str, choices=["tesseract_ocr", "easy_ocr"], required=True,
                    help="Which OCR engine subfolder to use under processed/ocr/")
    ap.add_argument("--level", type=str, choices=["level1", "level2", "level3"], required=True,
                    help="Scan level to use under processed/ocr/<engine>/<level>/docs/")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))
    ocr_root = ROOT / cfg["paths"]["data"]["processed"]["ocr_root"]
    docs_root = ocr_root / args.engine / args.level 

    model_dir = ROOT / cfg["paths"]["data"]["models"]["lr_doc_type_classifier"][args.engine]
    ensure_dir(model_dir)

    model_path = model_dir / f"doc_type_classifier_{args.level}.pkl"
    vect_path = model_dir / f"doc_type_vect_{args.level}.pkl"

    print(f"Engine    : {args.engine}")
    print(f"Level     : {args.level}")
    print(f"Docs root : {docs_root}")
    print(f"Model out : {model_path}")
    print(f"Vect  out : {vect_path}")

    # --- load texts + labels from folder names ---
    docs, labels, paths = collect_docs_and_labels(docs_root)

    print(f"\nLoaded {len(docs)} training documents.")
    if not docs:
        raise RuntimeError(f"No non-empty .txt files found under {docs_root}")

    dist = Counter(labels)
    print("Class distribution (all):", dict(dist))

    # --- TF‑IDF ---
    tfidf_cfg = cfg["doc_type_classifier"]["tfidf"]
    vec = TfidfVectorizer(
        stop_words=(tfidf_cfg.get("stop_words") or None),
        max_features=tfidf_cfg["max_features"],
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    print(f"\nFitting TF‑IDF (max_features={tfidf_cfg['max_features']}) ...")
    X = vec.fit_transform(docs)
    y = labels
    print(f"TF‑IDF shape: {X.shape}")

    # --- split ---
    split_cfg = cfg["doc_type_classifier"]["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        stratify=y if split_cfg.get("stratify") else None,
        random_state=split_cfg["random_state"]
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print("Class distribution (train):", dict(Counter(y_train)))
    print("Class distribution (test) :", dict(Counter(y_test)))

    # --- train ---
    lr_cfg = cfg["doc_type_classifier"]["logistic_regression"]
    print("\nTraining LogisticRegression with params:", lr_cfg)
    clf = LogisticRegression(**lr_cfg)
    clf.fit(X_train, y_train)
    print("Training complete.")

    # --- eval ---
    print("\nEvaluation on test set:")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # --- cross-val ---
    print("\nCross-validation (cv=5) on full data...")
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"CV scores: {cv_scores}")
    print(f"CV mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # --- retrain on all data and save ---
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    joblib.dump(vec, vect_path)
    print(f"\nSaved model   → {model_path}")
    print(f"Saved vector  → {vect_path}")

    # --- metrics JSON ---
    metrics = {
        "engine": args.engine,
        "level": args.level,
        "n_docs": len(docs),
        "n_classes": len(set(labels)),
        "classes": sorted(list(set(labels))),
        "accuracy_test": float(acc),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "tfidf_max_features": tfidf_cfg["max_features"],
        "lr_params": lr_cfg,
    }
    metrics_path = model_dir / f"doc_type_classifier_metrics_{args.level}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics → {metrics_path}")

    # --- show top features per class (optional, console) ---
    try:
        feature_names = vec.get_feature_names_out()
        for ci, cls in enumerate(clf.classes_):
            coefs = clf.coef_[ci]
            top = coefs.argsort()[-10:][::-1]
            print(f"\nTop features for '{cls}':")
            for idx in top:
                print(f"  {feature_names[idx]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
