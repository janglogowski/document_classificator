import argparse
import os
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import math
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

# ------------------------ Utilities ------------------------ #

def load_cfg(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def read_text_safe(path: Path) -> str:
    """Read UTF-8 text with replacement to avoid crashes on bad bytes."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def list_txt_recursive(root: Path) -> List[Path]:
    """Return all *.txt files under root (recursively)."""
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def sample_per_subfolder(folder: Path, k_per_sub: int) -> List[Path]:
    """
    For docs, we expect subfolders per type. Randomly sample up to k files from each subfolder.
    """
    out: List[Path] = []
    if not folder.exists():
        return out
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir():
            continue
        txts = list_txt_recursive(sub)
        if not txts:
            continue
        random.shuffle(txts)
        out.extend(txts[: min(k_per_sub, len(txts))])
    return out


def sample_uniform(paths: List[Path], k: int) -> List[Path]:
    """Randomly sample up to k items from a flat list."""
    if not paths:
        return []
    if len(paths) <= k:
        return list(paths)
    return random.sample(paths, k)


def balance_classes(doc_paths: List[Path], drw_paths: List[Path], max_per_class: int | None = None) -> Tuple[List[Path], List[Path]]:
    """
    Optionally downsample the larger class to keep classes balanced.
    If max_per_class is provided, additionally cap both classes to that size.
    """
    n_doc = len(doc_paths)
    n_drw = len(drw_paths)
    if n_doc == 0 or n_drw == 0:
        return doc_paths, drw_paths

    target = min(n_doc, n_drw)
    if max_per_class is not None:
        target = min(target, max_per_class)

    random.shuffle(doc_paths)
    random.shuffle(drw_paths)
    return doc_paths[:target], drw_paths[:target]


# ------------------------ Helpers for config paths ------------------------ #

def _dig(d, keys):
    """Safely navigate nested dict by a tuple of keys; return None if any is missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _abs_from_root(root: Path, p) -> Path:
    """Return absolute Path; if 'p' is relative, join with 'root'."""
    p = Path(p)
    return p if p.is_absolute() else (root / p)


# ------------------------ Main training ------------------------ #

def train_doc_vs_drw(
    config_path: Path,
    engine: str,
    level: str,
    k_docs_per_sub: int = 43,
    k_drawings_cap: int = 258,
    balance: bool = True
) -> None:
    """
    Train a Logistic Regression classifier (document vs. technical drawing) on OCR text.
    """

    cfg = load_cfg(config_path)

    # ROOT (required)
    project_root = _dig(cfg, ("paths", "project_root")) or cfg.get("project_root")
    if not project_root:
        raise KeyError("Missing 'paths.project_root' (or top-level 'project_root') in config.yaml")
    ROOT = Path(os.path.abspath(project_root))

    # OCR root (required) — prefer paths.data.processed.ocr_root
    ocr_root_cfg = (
        _dig(cfg, ("paths", "data", "processed", "ocr_root"))
        or _dig(cfg, ("paths", "processed", "ocr_root"))
        or _dig(cfg, ("data", "processed", "ocr_root"))
        or _dig(cfg, ("processed", "ocr_root"))
    )
    if not ocr_root_cfg:
        raise KeyError("Missing 'paths.data.processed.ocr_root' in config.yaml")
    ocr_root = _abs_from_root(ROOT, ocr_root_cfg)

    level_dir = ocr_root / engine / level
    docs_dir  = level_dir / "docs"
    docs_root = docs_dir if docs_dir.exists() else level_dir
    drw_root = ocr_root / engine / "technical_drawings"

    models_map = _dig(cfg, ("paths", "data", "models", "lr_doc_vs_drw"))
    models_base = _dig(cfg, ("paths", "models", "lr_doc_vs_drw"))

    if models_map and isinstance(models_map, dict):
        if engine not in models_map:
            raise KeyError(
                f"Missing path for engine='{engine}' under 'paths.data.models.lr_doc_vs_drw' in config.yaml"
            )
        model_dir = _abs_from_root(ROOT, models_map[engine])
    elif models_base and isinstance(models_base, (str, Path)):
        model_dir = _abs_from_root(ROOT, models_base) / engine
    else:
        raise KeyError("Provide valid model output path in config.yaml")

    ensure_dir(model_dir)
    model_path   = model_dir / f"lr_doc_vs_drw_{level}.pkl"
    vect_path    = model_dir / f"doc_vs_drw_vect_{level}.pkl"
    metrics_path = model_dir / f"doc_vs_drw_metrics_{level}.json"

    print(f"Engine          : {engine}")
    print(f"Level           : {level}")
    print(f"Docs root       : {docs_root}")
    print(f"Drawings root   : {drw_root}")
    print(f"Model out       : {model_path}")
    print(f"Vectorizer out  : {vect_path}")

    # Collect docs (k per subfolder) and drawings (cap)
    docs_paths = sample_per_subfolder(docs_root, k_per_sub=k_docs_per_sub)
    drw_paths  = sample_uniform(list_txt_recursive(drw_root), k=k_drawings_cap)

    print(f"Collected docs  : {len(docs_paths)}")
    print(f"Collected drw   : {len(drw_paths)}")

    if not docs_paths:
        raise FileNotFoundError(f"No document .txt files found under {docs_root}")
    if not drw_paths:
        raise FileNotFoundError(f"No drawing .txt files found under {drw_root}")

    if balance:
        docs_paths, drw_paths = balance_classes(docs_paths, drw_paths)
        print(f"After balancing : docs={len(docs_paths)} drw={len(drw_paths)}")

    texts, labels = [], []
    for p in docs_paths:
        t = read_text_safe(p)
        if t:
            texts.append(t)
            labels.append("document")
    for p in drw_paths:
        t = read_text_safe(p)
        if t:
            texts.append(t)
            labels.append("tech_drw")

    if not texts:
        raise RuntimeError("No non-empty texts to train on.")

    print("Label distribution:", dict(Counter(labels)))

    # TF–IDF
    tv_cfg = cfg["doc_vs_drw_classifier"]["tfidf"]
    vectorizer = TfidfVectorizer(
        stop_words=tv_cfg.get("stop_words") or None,
        max_features=tv_cfg["max_features"],
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    X = vectorizer.fit_transform(texts)
    y = labels
    print(f"TF–IDF matrix shape: {X.shape}")

    # Split
    split_cfg = cfg["doc_vs_drw_classifier"]["train_test_split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        stratify=y if split_cfg.get("stratify") else None,
        random_state=split_cfg["random_state"]
    )
    print(f"Train/Test sizes: {X_train.shape[0]} / {X_test.shape[0]}")
    print("Class distribution (train):", dict(Counter(y_train)))
    print("Class distribution (test) :", dict(Counter(y_test)))

    # Train LR
    lr_cfg = cfg["doc_vs_drw_classifier"]["logistic_regression"]
    model = LogisticRegression(**lr_cfg)
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    print("\n=== EVALUATION ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vect_path)
    print(f"\nSaved model      >>>> {model_path}")
    print(f"Saved vectorizer >>>> {vect_path}")

    # Save metrics
    classes_sorted = sorted(set(labels))
    dtc_like = cfg.get("doc_vs_drw_classifier", {})  # re-używamy tej samej sekcji
    cv_k  = int(dtc_like.get("cv_splits", 5))
    cv_rs = int(dtc_like.get("random_state", 41))
    holdout_enabled = bool(dtc_like.get("holdout_enabled", True))

    # Pipeline do CV na surowych tekstach (TFIDF + LR)
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

    skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=cv_rs)

    print(f"\nCross-validation (k={cv_k}, stratified, rs={cv_rs}) on raw texts ...")
    acc_scores = cross_val_score(pipe, texts, labels, cv=skf, scoring="accuracy")
    try:
        f1_scores = cross_val_score(pipe, texts, labels, cv=skf, scoring="f1_macro")
    except Exception:
        f1_scores = None

    # OOF predictions -> classification report
    y_pred_cv = cross_val_predict(pipe, texts, labels, cv=skf)
    cv_report = classification_report(labels, y_pred_cv, output_dict=True, labels=classes_sorted)

    # 95% CI dla accuracy
    def _tcrit_95(df: int) -> float:
        table = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447,
                 7:2.365, 8:2.306, 9:2.262, 10:2.228, 15:2.131, 20:2.086, 30:2.042}
        if df in table: return table[df]
        if df < 1: return 12.706
        return 1.96

    def _ci95_from_scores(scores: np.ndarray) -> tuple[float, float]:
        k = len(scores)
        mean = float(np.mean(scores)) if k else 0.0
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

    # Holdout już policzony wyżej (acc, rep, X_train/X_test, y_train/y_test)
    holdout_section = None
    if holdout_enabled:
        holdout_section = {
            "test_size": float(cfg["doc_vs_drw_classifier"]["train_test_split"]["test_size"]),
            "train_size": int(X_train.shape[0]),
            "test_size_abs": int(X_test.shape[0]),
            "accuracy": float(acc),
            "classification_report": rep,
        }

    metrics = {
        "engine": engine,
        "level": level,
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
        "holdout": holdout_section
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics    >>>> {metrics_path}")


# ------------------------ CLI ------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train LR doc-vs-drawing classifier on OCR text.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--engine", type=str, choices=["tesseract_ocr", "easy_ocr"], required=True)
    parser.add_argument("--level", type=str, choices=["level1", "level2", "level3"], required=True)
    parser.add_argument("--docs-per-sub", type=int, default=43, help="Max docs sampled per subfolder")
    parser.add_argument("--drawings-cap", type=int, default=258, help="Max drawings sampled in total")
    parser.add_argument("--no-balance", action="store_true", help="Disable class downsampling")
    args = parser.parse_args()

    train_doc_vs_drw(
        config_path=Path(args.config),
        engine=args.engine,
        level=args.level,
        k_docs_per_sub=args.docs_per_sub,
        k_drawings_cap=args.drawings_cap,
        balance=(not args.no_balance),
    )


if __name__ == "__main__":
    main()
