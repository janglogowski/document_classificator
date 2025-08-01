import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib


def load_cfg(config_path: Path) -> dict:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    """Create directory (parents included) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def read_ocr_table(tables_root: Path) -> pd.DataFrame:
    """Load OCR table from Parquet if available, else CSV."""
    p_parquet = tables_root / "ocr.parquet"
    p_csv = tables_root / "ocr.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"Missing ocr.parquet and ocr.csv in {tables_root}")


def read_labels(tables_root: Path) -> pd.DataFrame:
    """Load labels.csv created by build_tables.py."""
    p = tables_root / "labels.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run build_tables.py first.")
    return pd.read_csv(p)


def basic_text_stats(text: str) -> Dict[str, float]:
    """Compute simple text statistics for human-readable diagnostics."""
    toks = text.split()
    n = len(toks)
    chars = len(text)
    uniq = len(set(toks))
    avg_w = float(np.mean([len(t) for t in toks])) if n > 0 else 0.0
    return {"words": n, "chars": chars, "uniq_words": uniq, "avg_word_len": avg_w}


def topk_terms_for_row(row_idx: int, row_csr: sparse.csr_matrix, vocab_inv: List[str], k: int) -> List[Dict]:
    """Return top‑K TF‑IDF terms for a single row as a list of dicts."""
    start = row_csr.indptr[row_idx]
    end = row_csr.indptr[row_idx + 1]
    indices = row_csr.indices[start:end]
    data = row_csr.data[start:end]

    if data.size == 0:
        return []

    order = np.argsort(-data)
    top_idx = order[:k]
    out = []
    for j in top_idx:
        term_idx = indices[j]
        out.append({"term": vocab_inv[term_idx], "weight": float(data[j])})
    return out


# --------------------------- builder --------------------------- #

def build_features(
    config_path: Path,
    topk: int = 200,
    min_df: int = 2,
    max_df: float = 0.9,
    ngram: str = "1-2",
    stop_words: str = None,
    max_features: int = None
) -> None:
    """
    Build features for two tasks:
      1) doc_vs_drw (binary): uses ALL rows, y = is_document (0/1)
      2) doc_type (multiclass): uses ONLY documents, y = doc_type (label-encoded)
    """
    cfg = load_cfg(config_path)

    root = Path(os.path.abspath(cfg["paths"]["project_root"]))
    tables_root = root / cfg["paths"]["data"]["processed"]["tables_root"]
    ensure_dir(tables_root)

    # Load tables
    df_ocr = read_ocr_table(tables_root)
    df_lbl = read_labels(tables_root)

    # Merge to have text + labels
    df = df_ocr.merge(df_lbl, on="path", how="left")

    # Clean up and guard against missing/NaN
    df["text"] = df["text"].astype(str).fillna("")
    df["is_document"] = df["is_document"].fillna(False).astype(bool)
    df["doc_type"] = df["doc_type"].fillna("unknown").astype(str)

    # Vectorizer config
    ngram_range = tuple(int(x) for x in ngram.split("-"))
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words=stop_words,
    )

    # Fit on ALL texts once → consistent features for both tasks
    X_all = vectorizer.fit_transform(df["text"].astype(str))

    # Save vectorizer
    vec_path = tables_root / "vectorizer.joblib"
    joblib.dump(vectorizer, vec_path)
    print(f"[OK] Saved vectorizer: {vec_path}")

    # ---------------- doc_vs_drw (ALL rows) ---------------- #
    y_docvsdrw = df["is_document"].astype(int).values  # 1 = document, 0 = drawing
    X_docvsdrw = X_all

    sparse.save_npz(tables_root / "X_docvsdrw.npz", X_docvsdrw)
    np.save(tables_root / "y_docvsdrw.npy", y_docvsdrw)
    df[["path", "is_document"]].to_csv(tables_root / "index_docvsdrw.csv", index=False, encoding="utf-8")
    print(f"[OK] Saved doc_vs_drw: X={X_docvsdrw.shape}, y={y_docvsdrw.shape}")

    # ---------------- doc_type (DOCUMENTS only) ---------------- #
    docs_mask = df["is_document"] == True
    docs = df[docs_mask].copy()

    if docs.empty:
        print("[INFO] No documents available to build doc_type features.")
    else:
        X_docs_type = vectorizer.transform(docs["text"].astype(str))

        # Label-encode doc_type
        le = LabelEncoder()
        y_docs_type = le.fit_transform(docs["doc_type"].astype(str))

        sparse.save_npz(tables_root / "X_docs_type.npz", X_docs_type)
        np.save(tables_root / "y_docs_type.npy", y_docs_type)
        docs[["path", "doc_type"]].to_csv(tables_root / "index_docs_type.csv", index=False, encoding="utf-8")
        joblib.dump(le, tables_root / "doc_type_le.joblib")

        print(f"[OK] Saved doc_type: X={X_docs_type.shape}, y={y_docs_type.shape}")
        print(f"[OK] Saved doc_type label encoder: {tables_root / 'doc_type_le.joblib'}")

        try:
            vocab = vectorizer.vocabulary_
            vocab_inv = [""] * len(vocab)
            for term, idx in vocab.items():
                vocab_inv[idx] = term

            feats_rows = []
            for i, (_, row) in enumerate(docs.reset_index(drop=True).iterrows()):
                stats = basic_text_stats(str(row["text"]))
                top_terms = topk_terms_for_row(i, X_docs_type, vocab_inv, topk)
                feats_rows.append(
                    {
                        "path": row["path"],
                        "doc_type": row.get("doc_type", None),
                        **stats,
                        "tfidf_top_terms": json.dumps(top_terms, ensure_ascii=False),
                    }
                )
            df_features = pd.DataFrame(feats_rows)
            feat_parquet = tables_root / "features_docs.parquet"
            try:
                df_features.to_parquet(feat_parquet, index=False)
                print(f"[OK] Saved: {feat_parquet}")
            except Exception as e:
                fallback = tables_root / "features_docs.csv"
                print(f"[WARN] Parquet not available ({e}). Writing CSV fallback.")
                df_features.to_csv(fallback, index=False, encoding="utf-8")
                print(f"[OK] Saved: {fallback}")

        except Exception as e:
            print(f"[WARN] Skipped human-readable features build due to: {e}")

    print("\n[SUMMARY]")
    try:
        print(df.groupby(["is_document", "doc_type"]).size())
    except Exception:
        pass


# --------------------------- CLI --------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Build TF‑IDF features for doc_vs_drw and doc_type tasks.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--topk", type=int, default=200, help="Number of top‑K TF‑IDF terms to store per doc (diagnostics)")
    parser.add_argument("--min-df", type=int, default=2, help="min_df for TF‑IDF")
    parser.add_argument("--max-df", type=float, default=0.9, help="max_df for TF‑IDF")
    parser.add_argument("--ngram", type=str, default="1-2", help="ngram range, e.g. '1-2' or '1-3'")
    parser.add_argument("--stop-words", type=str, default=None, help="Use 'english' or leave empty")
    parser.add_argument("--max-features", type=int, default=None, help="Cap the vocabulary size (optional)")
    args = parser.parse_args()

    build_features(
        config_path=Path(args.config),
        topk=args.topk,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram=args.ngram,
        stop_words=args.stop_words,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
