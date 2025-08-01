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
    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))

    ocr_root = ROOT / cfg["paths"]["data"]["processed"]["ocr_root"]
    docs_root = ocr_root / engine / level                    # data/processed/ocr/<engine>/<level>/docs/<type>/*.txt
    drw_root  = ocr_root / engine / "technical_drawings"     # data/processed/ocr/<engine>/technical_drawings/**/*.txt

    # outputs
    model_dir = ROOT / cfg["paths"]["data"]["models"]["lr_doc_vs_drw"] / engine
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

    # Balance classes (optional but recommended for LR)
    if balance:
        docs_paths, drw_paths = balance_classes(docs_paths, drw_paths)
        print(f"After balancing : docs={len(docs_paths)} drw={len(drw_paths)}")

    # Read texts + labels
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
    metrics = {
        "engine": engine,
        "level": level,
        "n_docs": len(docs_paths),
        "n_drawings": len(drw_paths),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "accuracy": acc,
        "classification_report": rep,
        "tfidf_max_features": tv_cfg["max_features"],
        "logistic_regression_params": lr_cfg
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
