import argparse
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import yaml


def load_cfg(config_path: Path) -> dict:
    """Load YAML configuration from disk."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    """Create directory (parents included) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    """Read UTF‑8 text from file; return empty string on error."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def discover_txt(ocr_root: Path) -> List[Path]:
    """Return a recursive list of all .txt files under the OCR root."""
    return [p for p in ocr_root.rglob("*.txt") if p.is_file()]


LEVEL_RE = re.compile(r"^level[123]$", re.IGNORECASE)

def normalize_token(s: str) -> str:
    """Normalize a path token for comparisons (upper for families, lower for keywords)."""
    return s.strip()

def _known_families_from_cfg(cfg: dict) -> Tuple[set, dict]:
    """
    Build a set of known doc families (e.g., BOMS, DAILY_REPORTS, …) and a map for case‑insensitive matching.
    """
    fams = set()
    fam_map = {}
    try:
        gens = cfg["paths"]["data"]["generators"]

        for k, v in gens.items():
            fams.add(v)
            fam_map[v.lower()] = v
    except Exception:
        pass
    if not fams:
        defaults = ["BOMS", "DAILY_REPORTS", "INSPECTION_REPORTS", "MAINTENANCE_LOGS", "PRODUCT_DATA_SHEETS", "QUALITY_CHECKLISTS"]
        for v in defaults:
            fams.add(v)
            fam_map[v.lower()] = v
    return fams, fam_map


def infer_metadata(txt_path: Path, ocr_root: Path, cfg: dict) -> Dict:
    rel = txt_path.relative_to(ocr_root)
    parts = list(rel.parts)

    engine = parts[0] if len(parts) >= 1 else "unknown"

    lower_parts = [p.lower() for p in parts]
    if "technical_drawings" in lower_parts:
        return {"engine": engine, "level": "all", "doc_family": "technical_drawings"}
    
    fams, fam_map = _known_families_from_cfg(cfg)


    level = "unknown"
    level_idx = None
    for i, tok in enumerate(parts):
        if LEVEL_RE.match(tok):
            level = tok
            level_idx = i
            break

    if "docs" in lower_parts:
        docs_idx = lower_parts.index("docs")
        if docs_idx + 1 < len(parts):
            cand = parts[docs_idx + 1]
            cand_norm = fam_map.get(cand.lower())
            if cand_norm:
                return {"engine": engine, "level": level, "doc_family": cand_norm}
            else:
                return {"engine": engine, "level": level, "doc_family": cand}

    if level_idx is not None:
        for j in range(level_idx + 1, len(parts)):
            cand = parts[j]
            cand_norm = fam_map.get(cand.lower())
            if cand_norm:
                return {"engine": engine, "level": level, "doc_family": cand_norm}
        if level_idx + 1 < len(parts):
            return {"engine": engine, "level": level, "doc_family": parts[level_idx + 1]}

    for cand in parts:
        cand_norm = fam_map.get(cand.lower())
        if cand_norm:
            return {"engine": engine, "level": level, "doc_family": cand_norm}

    return {"engine": engine, "level": level, "doc_family": "unknown"}


# --------------------------- main builder --------------------------- #

def build_tables(config_path: Path) -> None:
    """Scan processed OCR files and build ocr.parquet (or CSV) and labels.csv."""
    cfg = load_cfg(config_path)

    root = Path(os.path.abspath(cfg["paths"]["project_root"]))
    ocr_root = root / cfg["paths"]["data"]["processed"]["ocr_root"]
    tables_root = root / cfg["paths"]["data"]["processed"]["tables_root"]

    ensure_dir(tables_root)

    all_txt = discover_txt(ocr_root)
    if not all_txt:
        print(f"[INFO] No *.txt files found under {ocr_root}")
        return

    rows = []
    for p in all_txt:
        meta = infer_metadata(p, ocr_root, cfg)
        text = read_text(p)
        words = len(text.split())
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        except Exception:
            mtime = None

        rows.append(
            {
                "path": str(p.as_posix()),
                "engine": meta["engine"],
                "level": meta["level"],
                "doc_family": meta["doc_family"],
                "text": text,
                "words": words,
                "source_path": str(p.as_posix()),
                "source_mtime": mtime,
            }
        )

    df = pd.DataFrame(rows)

    ocr_parquet = tables_root / "ocr.parquet"
    ocr_csv = tables_root / "ocr.csv"
    try:
        df.to_parquet(ocr_parquet, index=False)
        print(f"[OK] Saved: {ocr_parquet}")
    except Exception as e:
        print(f"[WARN] Parquet unavailable ({e}). Saving CSV instead.")
        df.to_csv(ocr_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved: {ocr_csv}")

    labels = []
    for _, r in df.iterrows():
        is_drawing = (str(r["doc_family"]).lower() == "technical_drawings")
        labels.append(
            {
                "path": r["path"],
                "is_document": not is_drawing,
                "doc_type": (r["doc_family"] or "unknown"),
            }
        )

    labels_df = pd.DataFrame(labels).drop_duplicates(subset=["path"])
    labels_csv = tables_root / "labels.csv"
    labels_df.to_csv(labels_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved: {labels_csv}")

    try:
        print("\n[SUMMARY] counts by (engine, level, doc_family):")
        print(df.groupby(["engine", "level", "doc_family"]).size())
    except Exception:
        pass

    unknown = df[df["doc_family"] == "unknown"].head(10)
    if not unknown.empty:
        print("\n[DIAG] Examples that parsed as doc_family=unknown (showing up to 10):")
        for p in unknown["path"].tolist():
            rel = Path(p).relative_to(ocr_root)
            print("  -", rel)


# --------------------------- CLI --------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Build ocr.parquet (or CSV) and labels.csv from processed OCR outputs."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    build_tables(Path(args.config))


if __name__ == "__main__":
    main()

