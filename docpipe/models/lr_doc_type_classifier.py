import argparse
import os
import json
from collections import Counter
from pathlib import Path

import math
import joblib
import numpy as np
import yaml

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score, cross_val_predict)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------------ IO helpers ------------------------------ #

def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _models_root_for(cfg: dict, key: str, engine: str | None = None) -> Path:
    """
    Resolve models dir from config at paths.data.models.<key>.
    Supports string or dict (per-engine).
    """
    models_cfg = cfg["paths"]["data"]["models"][key]
    if isinstance(models_cfg, dict):
        if engine is None:
            raise KeyError(f"Model path for '{key}' requires engine; got None")
        base_rel = models_cfg.get(engine)
        if base_rel is None:
            raise KeyError(f"Missing models path for engine='{engine}' under paths.data.models.{key}")
    else:
        base_rel = models_cfg
    return Path(cfg["paths"]["project_root"]) / base_rel


# ------------------------------ Canonical labels ------------------------------ #

_CANON = {
    "BOM": "BOM", "BOMS": "BOM",
    "BILL_OF_MATERIAL": "BOM", "BILL_OF_MATERIALS": "BOM",
    "BILL OF MATERIAL": "BOM", "BILL OF MATERIALS": "BOM",

    "DAILY_REPORT": "DAILY_REPORT", "DAILY_REPORTS": "DAILY_REPORT",

    "INSPECTION_REPORT": "INSPECTION_REPORT", "INSPECTION_REPORTS": "INSPECTION_REPORT",

    "MAINTENANCE_LOG": "MAINTENANCE_LOG", "MAINTENANCE_LOGS": "MAINTENANCE_LOG",

    "PRODUCT_DATA_SHEET": "PRODUCT_DATA_SHEET", "PRODUCT_DATA_SHEETS": "PRODUCT_DATA_SHEET",
    "DATA_SHEET": "PRODUCT_DATA_SHEET", "DATASHEET": "PRODUCT_DATA_SHEET", "PDS": "PRODUCT_DATA_SHEET",

    "QUALITY_CHECKLIST": "QUALITY_CHECKLIST", "QUALITY_CHECKLISTS": "QUALITY_CHECKLIST", "QC": "QUALITY_CHECKLIST",
}

def _canon(label: str) -> str:
    lab = (label or "").strip().upper().replace("-", "_").replace(" ", "_")
    if not lab:
        return "UNDEFINED"
    if lab in _CANON:
        return _CANON[lab]
    # heurystyka: BILL + MATERIAL -> BOM
    if "BILL" in lab.replace("_", " ") and "MATERIAL" in lab.replace("_", " "):
        return "BOM"
    return lab


# ------------------------------ Data loading ------------------------------ #

def collect_docs_and_labels(docs_root: Path) -> tuple[list[str], list[str], list[Path]]:
    """
    Collect (text, label, path) from:
        docs_root/<LABEL>/**/*.txt
    where <LABEL> is canonized (BOM, DAILY_REPORT, ...).
    """
    texts: list[str] = []
    labels: list[str] = []
    paths: list[Path] = []

    if not docs_root.exists():
        raise FileNotFoundError(f"Docs root does not exist: {docs_root}")

    for class_dir in sorted([d for d in docs_root.iterdir() if d.is_dir()]):
        label = _canon(class_dir.name)
        for p in class_dir.rglob("*.txt"):
            if not p.is_file():
                continue
            text = read_text(p)
            if not text:
                continue
            texts.append(text)
            labels.append(label)
            paths.append(p)

    return texts, labels, paths


# ------------------------------ Main training ------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Train LR doc-type classifier from OCR text (unified CV metrics).")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--engine", type=str, choices=["tesseract_ocr", "easy_ocr"], required=True,
                    help="Which OCR engine subfolder to use under processed/ocr/")
    ap.add_argument("--level", type=str, choices=["level1", "level2", "level3"], required=True,
                    help="Scan level to use under processed/ocr/<engine>/<level>/")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))
    ocr_root = ROOT / cfg["paths"]["data"]["processed"]["ocr_root"]
    docs_root = ocr_root / args.engine / args.level

    model_dir = _models_root_for(cfg, "lr_doc_type_classifier", engine=args.engine)
    ensure_dir(model_dir)

    model_path = model_dir / f"lr_doc_type_{args.level}.pkl"
    vect_path  = model_dir / f"doc_type_vect_{args.level}.pkl"

    print(f"Engine    : {args.engine}")
    print(f"Level     : {args.level}")
    print(f"Docs root : {docs_root}")
    print(f"Model out : {model_path}")
    print(f"Vect  out : {vect_path}")

    # --- load texts + labels from folder names (canon) ---
    texts, labels, paths = collect_docs_and_labels(docs_root)
    if not texts:
        raise RuntimeError(f"No non-empty .txt files found under {docs_root}")
    print(f"\nLoaded {len(texts)} documents.")
    print("Class distribution (all):", dict(Counter(labels)))

    # --- config ---
    dtc = cfg["doc_type_classifier"]
    tv_cfg = dtc["tfidf"]
    lr_cfg = dtc["logistic_regression"]

    cv_k  = int(dtc.get("cv_splits", 5))
    cv_rs = int(dtc.get("random_state", 41))
    split_cfg = dtc.get("train_test_split", {"test_size": 0.2, "random_state": 41, "stratify": True})

    # --- pipeline for CV ---
    pipe = make_pipeline(
        TfidfVectorizer(
            stop_words=(tv_cfg.get("stop_words") or None),
            max_features=int(tv_cfg["max_features"]),
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
        ),
        LogisticRegression(**lr_cfg),
    )

    # --- Stratified CV ---
    skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=cv_rs)

    print(f"\nCross-validation (k={cv_k}, stratified, rs={cv_rs}) on raw texts ...")
    acc_scores = cross_val_score(pipe, texts, labels, cv=skf, scoring="accuracy")
    try:
        f1_scores = cross_val_score(pipe, texts, labels, cv=skf, scoring="f1_macro")
    except Exception:
        f1_scores = None

    # OOF predictions for classification report
    y_pred_cv = cross_val_predict(pipe, texts, labels, cv=skf)
    cv_report = classification_report(labels, y_pred_cv, output_dict=True, labels=sorted(set(labels)))

    # --- Final (CV) summary: mean, std, 95% CI ---
    def _tcrit_95(df: int) -> float:
        table = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447,
                 7:2.365, 8:2.306, 9:2.262, 10:2.228, 15:2.131, 20:2.086, 30:2.042}
        if df in table: return table[df]
        if df < 1: return 12.706
        return 1.96

    def _ci95_from_scores(scores: np.ndarray) -> tuple[float, float]:
        k = len(scores)
        mean = float(np.mean(scores))
        std_sample = float(np.std(scores, ddof=1)) if k > 1 else 0.0
        se = std_sample / math.sqrt(k) if k > 0 else 0.0
        tcrit = _tcrit_95(k - 1)
        lo = max(0.0, mean - tcrit * se)
        hi = min(1.0, mean + tcrit * se)
        return lo, hi

    cv_mean = float(np.mean(acc_scores))
    cv_std  = float(np.std(acc_scores, ddof=0))
    ci_lo, ci_hi = _ci95_from_scores(acc_scores)

    f1_mean = float(np.mean(f1_scores)) if f1_scores is not None else None
    f1_std  = float(np.std(f1_scores, ddof=0)) if f1_scores is not None else None

    print(f"CV accuracy: {acc_scores}")
    print(f"CV mean: {cv_mean:.4f} ± {cv_std:.4f}   |  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    if f1_scores is not None:
        print(f"CV f1_macro: {f1_scores}")
        print(f"CV f1 mean: {f1_mean:.4f} ± {f1_std:.4f}")

    # --- Optional holdout ---
    holdout_enabled = bool(dtc.get("holdout_enabled", True))
    if holdout_enabled:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=split_cfg.get("test_size", 0.2),
            stratify=labels if split_cfg.get("stratify", True) else None,
            random_state=split_cfg.get("random_state", 41),
        )
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred_test)
        test_report = classification_report(y_test, y_pred_test, output_dict=True, labels=sorted(set(labels)))
    else:
        acc_test = None
        test_report = None

    # --- Train final vectorizer + LR on ALL data for compatibility with existing loaders ---
    vec = TfidfVectorizer(
        stop_words=(tv_cfg.get("stop_words") or None),
        max_features=int(tv_cfg["max_features"]),
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    X_all = vec.fit_transform(texts)
    clf = LogisticRegression(**lr_cfg).fit(X_all, labels)

    joblib.dump(clf, model_path)
    joblib.dump(vec, vect_path)

    print(f"\nSaved model   >>>> {model_path}")
    print(f"Saved vector  >>>> {vect_path}")

    # --- Unified metrics JSON ---
    classes_sorted = sorted(set(labels))
    metrics = {
        "engine": args.engine,
        "level": args.level,
        "n_docs": len(texts),
        "n_classes": len(classes_sorted),
        "classes": classes_sorted,
        "vectorizer": {
            "max_features": int(tv_cfg["max_features"]),
            "stop_words": tv_cfg.get("stop_words"),
            "ngram_range": [1, 2],
            "strip_accents": "unicode",
            "lowercase": True,
        },
        "model": {
            "name": "LogisticRegression",
            "params": lr_cfg,
        },
        "final": {
            "accuracy": cv_mean,
            "accuracy_ci95": [ci_lo, ci_hi],
            **({"f1_macro": f1_mean} if f1_mean is not None else {}),
            "origin": {"type": "cv_mean", "k": cv_k}
        },
        "cv": {
            "k": cv_k,
            "random_state": cv_rs,
            "accuracy_scores": [float(x) for x in acc_scores],
            "accuracy_mean": cv_mean,
            "accuracy_std": cv_std,
            **({
                "f1_macro_scores": [float(x) for x in f1_scores],
                "f1_macro_mean": f1_mean,
                "f1_macro_std": f1_std,
            } if f1_scores is not None else {}),
            "classification_report": cv_report,
        },
        "holdout": (None if not holdout_enabled else {
            "test_size": float(split_cfg.get("test_size", 0.2)),
            "train_size": int(len(X_train)),
            "test_size_abs": int(len(X_test)),
            "accuracy": float(acc_test),
            "classification_report": test_report,
        })
    }

    metrics_path = model_dir / f"doc_type_classifier_metrics_{args.level}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics >>>> {metrics_path}")

    # --- Top features per class ---
    try:
        feature_names = vec.get_feature_names_out()
        for ci, cls in enumerate(clf.classes_):
            coefs = clf.coef_[ci]
            top_idx = coefs.argsort()[-10:][::-1]
            print(f"\nTop features for '{cls}':")
            for idx in top_idx:
                print(f"  {feature_names[idx]}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
