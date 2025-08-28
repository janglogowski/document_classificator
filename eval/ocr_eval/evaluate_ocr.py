# eval/ocr_eval/evaluate_ocr.py
from __future__ import annotations

import sys
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
import unicodedata
from collections import Counter
import numpy as np
import pandas as pd
import yaml

THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parents[2] 
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from docpipe.ocr.easyocr_engine import EasyOCREngine
from docpipe.ocr.tesseract_engine import TesseractOCREngine

# --- BoW (case-insensitive) ---
_BOW_RE = re.compile(r"\w+", re.UNICODE)  # słowa alfanumiczne (z diakrytykami)

def bow_tokens(s: str) -> List[str]:
    s = normalize_text(s, lower=True, strip=True)
    toks = _BOW_RE.findall(s)
    return [t for t in toks if t != "_"]

def bow_metrics(gt: str, pred: str, mode: str = "multiset") -> Tuple[float, float, float]:
    """
    Return (recall, precision, f1) for bag-of-words.
    - mode="multiset": Counter
    - mode="unique":   Unique words
    """
    gt_tokens = bow_tokens(gt)
    pr_tokens = bow_tokens(pred)

    if not gt_tokens and not pr_tokens:
        return 1.0, 1.0, 1.0
    if not gt_tokens:
        p = 1.0 if not pr_tokens else 0.0
        f = 0.0 if p == 0 else 1.0
        return 1.0, p, f
    if not pr_tokens:
        return 0.0, 0.0, 0.0

    if mode == "unique":
        gt_set, pr_set = set(gt_tokens), set(pr_tokens)
        hits = len(gt_set & pr_set)
        rec = hits / max(1, len(gt_set))
        prec = hits / max(1, len(pr_set))
    else:
        gt_c, pr_c = Counter(gt_tokens), Counter(pr_tokens)
        hits = sum(min(gt_c[w], pr_c[w]) for w in gt_c.keys() | pr_c.keys())
        rec = hits / max(1, sum(gt_c.values()))
        prec = hits / max(1, sum(pr_c.values()))

    f1 = 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)
    return rec, prec, f1

# -------------------- utils: normalizacja, metryki -------------------- #

_WORD_RE = re.compile(r"\w+|\S", re.UNICODE)

def normalize_text(s: str, lower: bool = True, strip: bool = True) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "").replace("\ufeff", "")
    if strip:
        s = s.strip()
    if lower:
        s = s.lower()
    return s

def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, :] = np.arange(m + 1)
    dp[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,          # delete
                dp[i, j - 1] + 1,          # insert
                dp[i - 1, j - 1] + cost    # substitute
            )
    return int(dp[n, m])

def cer_and_acc(gt: str, pred: str, keep: str = "alnum") -> tuple[float, float]:
    """
    keep:
      - 'alnum' CER tylko po znakach [0-9a-z] (layout-invariant)
      - 'all'   klasyczny CER po wszystkich znakach
    """
    gt_norm = normalize_text(gt)
    pr_norm = normalize_text(pred)

    if keep == "alnum":
        gt_chars = re.findall(r"[0-9a-z]", gt_norm, flags=re.UNICODE)
        pr_chars = re.findall(r"[0-9a-z]", pr_norm, flags=re.UNICODE)
    else:
        gt_chars = list(gt_norm)
        pr_chars = list(pr_norm)

    if len(gt_chars) == 0:
        cer = 0.0 if len(pr_chars) == 0 else 1.0
        return cer, 1.0 - cer

    dist = levenshtein(gt_chars, pr_chars)
    cer = dist / max(1, len(gt_chars))
    return cer, 1.0 - cer

def tokenize_words(s: str, keep_punct: bool) -> List[str]:
    if keep_punct:
        return re.findall(r"\w+|\S", s) 
    else:
        return re.findall(r"\w+", s)

def wer(gt: str, pred: str, mode: str = "nopunct") -> float:
    """
    mode:
      - 'nopunct' tylko słowa/numery (ignoruj interpunkcję i layout)
      - 'strict'  klasyczny WER z interpunkcją
    """
    gt_norm = normalize_text(gt)
    pr_norm = normalize_text(pred)

    if mode == "nopunct":
        gt_words = re.findall(r"[0-9a-z]+", gt_norm, flags=re.UNICODE)
        pr_words = re.findall(r"[0-9a-z]+", pr_norm, flags=re.UNICODE)
    else:
        gt_words = _WORD_RE.findall(gt_norm)
        pr_words = _WORD_RE.findall(pr_norm)

    if len(gt_words) == 0:
        return 0.0 if len(pr_words) == 0 else 1.0
    dist = levenshtein(gt_words, pr_words)
    return dist / max(1, len(gt_words))


# -------------------- helpers: IO -------------------- #

ID_PAT = re.compile(r"^doc_(\d+)_level\d+(?:_p\d+)?\.(?:png|jpg|jpeg)$", re.IGNORECASE)

def load_gt_map(gt_dir: Path) -> Dict[str, str]:
    """Zwraca mapę: doc_id (str) -> ground truth text."""
    gt_map: Dict[str, str] = {}
    for p in gt_dir.glob("doc_*.gt.txt"):
        m = re.match(r"^doc_(\d+)\.gt\.txt$", p.name)
        if m:
            gt_map[m.group(1)] = p.read_text(encoding="utf-8", errors="ignore")
    return gt_map

def list_level_images(level_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(level_dir.glob(ext))
    return sorted(imgs)

def extract_id_from_image_name(name: str) -> str:
    m = ID_PAT.match(name)
    return m.group(1) if m else ""

def eval_engine_on_levels(
    scans_root: Path,
    gt_dir: Path,
    engines: List[str],
    levels: List[str],
    out_dir: Path,
    cfg_path: Path,
    wer_mode: str = "nopunct",
    cer_keep: str = "alnum",
) -> None:
    
    keep_punct = (wer_mode == "strict")       
    alnum_only = (cer_keep == "alnum")       

    def _cer_and_acc(gt: str, pred: str) -> Tuple[float, float]:
        g = normalize_text(gt)
        p = normalize_text(pred)
        if alnum_only:
            g = re.sub(r"[^0-9a-z ]+", "", g)
            p = re.sub(r"[^0-9a-z ]+", "", p)
        g_chars = list(g)
        p_chars = list(p)
        if len(g_chars) == 0:
            cer = 0.0 if len(p_chars) == 0 else 1.0
            return cer, 1.0 - cer
        dist = levenshtein(g_chars, p_chars)
        cer = dist / max(1, len(g_chars))
        return cer, 1.0 - cer

    def _wer(gt: str, pred: str) -> float:
        g = normalize_text(gt)
        p = normalize_text(pred)
        if keep_punct:
            g_tokens = _WORD_RE.findall(g)  
            p_tokens = _WORD_RE.findall(p)
        else:
            g_tokens = re.findall(r"\w+", g)
            p_tokens = re.findall(r"\w+", p)
        if len(g_tokens) == 0:
            return 0.0 if len(p_tokens) == 0 else 1.0
        dist = levenshtein(g_tokens, p_tokens)
        return dist / max(1, len(g_tokens))

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_map = load_gt_map(gt_dir)

    manifest_map: Dict[str, str] = {}
    manifest_path = gt_dir.parent / "manifest.csv"
    if manifest_path.exists():
        df_manifest = pd.read_csv(manifest_path)
        for _, row in df_manifest.iterrows():
            manifest_map[str(row["doc_id"])] = row["doc_type"]

    # config + poppler
    if not cfg_path.exists():
        raise SystemExit(f"[ERR] Brak pliku config.yaml pod ścieżką: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    poppler_path = (((cfg.get("paths") or {}).get("poppler") or {}).get("bin_path"))

    engine_objs: Dict[str, object] = {}
    for eng in set(e.lower() for e in engines):
        if eng == "tesseract":
            engine_objs["tesseract"] = TesseractOCREngine(cfg, poppler_path)
        elif eng == "easyocr":
            engine_objs["easyocr"] = EasyOCREngine(cfg, poppler_path)
        else:
            raise SystemExit("Dozwolone wartości --engines: tesseract,easyocr")

    for e in engines:
        if e.lower() not in engine_objs:
            raise SystemExit(f"[ERR] Nie udało się zainicjować silnika: {e}")

    all_rows: List[dict] = []
    summaries: List[dict] = []

    for level in levels:
        level = level.strip()
        if not level:
            continue

        level_dir = scans_root / level
        if not level_dir.exists():
            print(f"[WARN] Brak katalogu level: {level_dir}")
            continue

        images = list_level_images(level_dir)
        if not images:
            print(f"[WARN] Brak obrazów w: {level_dir}")
            continue

        for engine in engines:
            eng_key = engine.lower()
            per_rows: List[dict] = []

            for img in images:
                doc_id = extract_id_from_image_name(img.name)
                if not doc_id:
                    continue
                gt = gt_map.get(doc_id, "")
                if gt == "":
                    continue

                t0 = time.time()
                try:
                    res = engine_objs[eng_key].ocr_path(img)  
                    text = res[0] if isinstance(res, tuple) else str(res)
                except Exception as e:
                    print(f"[ERR] OCR '{eng_key}' padł na {img.name}: {e}")
                    text = ""
                dt = time.time() - t0

                cer, acc = cer_and_acc(gt, text, keep=cer_keep)
                w = wer(gt, text, mode=wer_mode)            
                bow_rec, bow_prec, bow_f1 = bow_metrics(gt, text, mode="multiset")

                per_rows.append({
                    "engine": eng_key,
                    "level": level,
                    "image": img.name,
                    "doc_id": doc_id,
                    "doc_type": manifest_map.get(doc_id, "unknown"),
                    "char_error_rate": cer,
                    "char_accuracy": acc,
                    "word_error_rate": w,
                    "time_sec": dt,
                    "bow_recall": bow_rec,
                    "bow_precision": bow_prec,
                    "bow_f1": bow_f1,
                })

            if per_rows:
                df_per = pd.DataFrame(per_rows).sort_values(["doc_id", "image"])
                df_per.to_csv(out_dir / f"per_file_{eng_key}_{level}.csv", index=False)

                cer_arr = df_per["char_error_rate"].to_numpy()
                wer_arr = df_per["word_error_rate"].to_numpy()
                t_arr   = df_per["time_sec"].to_numpy()

                summary = {
                    "engine": eng_key,
                    "level": level,
                    "n_files": int(len(df_per)),
                    "cer_mean": float(np.mean(cer_arr)),
                    "cer_median": float(np.median(cer_arr)),
                    "cer_p95": float(np.percentile(cer_arr, 95)),
                    "wer_mean": float(np.mean(wer_arr)),
                    "wer_median": float(np.median(wer_arr)),
                    "wer_p95": float(np.percentile(wer_arr, 95)),
                    "char_acc_mean": float(1.0 - np.mean(cer_arr)),
                    "char_acc_median": float(1.0 - np.median(cer_arr)),
                    "word_acc_mean": float(1.0 - np.mean(wer_arr)),
                    "word_acc_median": float(1.0 - np.median(wer_arr)),
                    "bow_recall_mean": float(np.mean(df_per["bow_recall"])),
                    "bow_recall_median": float(np.median(df_per["bow_recall"])),
                    "bow_precision_mean": float(np.mean(df_per["bow_precision"])),
                    "bow_precision_median": float(np.median(df_per["bow_precision"])),
                    "bow_f1_mean": float(np.mean(df_per["bow_f1"])),
                    "bow_f1_median": float(np.median(df_per["bow_f1"])),
                    "time_mean_sec": float(np.mean(t_arr)),
                    "time_median_sec": float(np.median(t_arr)),
                }
                summaries.append(summary)
                all_rows.extend(per_rows)

                print(
                    f"[{eng_key}][{level}] "
                    f"n={summary['n_files']}  "
                    f"CER(mean)={summary['cer_mean']:.3f}  "
                    f"WER(mean)={summary['wer_mean']:.3f}  "
                    f"content_recall={summary['bow_recall_mean']:.3f}  "
                    f"content_f1={summary['bow_f1_mean']:.3f}  "
                    f"t_mean={summary['time_mean_sec']:.3f}s"
                )
    if not all_rows:
        print("[INFO] Nothing saved.")
        return

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out_dir / "per_file_all.csv", index=False)

    # summary per (engine, level, doc_type)
    df_group = df_all.groupby(["engine", "level", "doc_type"]).agg({
        "char_error_rate": ["mean", "median"],
        "word_error_rate": ["mean", "median"],
        "bow_recall": ["mean", "median"],
        "bow_precision": ["mean", "median"],
        "bow_f1": ["mean", "median"],
        "time_sec": "mean"
    }).reset_index()

    df_group.columns = [
        "engine","level","doc_type",
        "cer_mean","cer_median",
        "wer_mean","wer_median",
        "bow_recall_mean","bow_recall_median",
        "bow_precision_mean","bow_precision_median",
        "bow_f1_mean","bow_f1_median",
        "time_mean_sec"
    ]
    df_group.to_csv(out_dir / "summary_by_class.csv", index=False)

    # summary per (engine, level)
    df_sum = df_all.groupby(["engine", "level"]).agg({
        "char_error_rate": ["mean", "median"],
        "word_error_rate": ["mean", "median"],
        "bow_recall": ["mean", "median"],
        "bow_precision": ["mean", "median"],
        "bow_f1": ["mean", "median"],
        "time_sec": "mean"
    }).reset_index()
    df_sum.columns = [
        "engine","level",
        "cer_mean","cer_median",
        "wer_mean","wer_median",
        "bow_recall_mean","bow_recall_median",
        "bow_precision_mean","bow_precision_median",
        "bow_f1_mean","bow_f1_median",
        "time_mean_sec"
    ]
    df_sum.to_csv(out_dir / "summary_by_level_engine.csv", index=False)

    print("\n=== SUMMARY BY CLASS ===")
    print(df_group.to_string(index=False))
    print("\n=== SUMMARY BY LEVEL & ENGINE ===")
    print(df_sum.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate OCR (project engines only) on eval set.")
    ap.add_argument("--scans_root", type=str, default="eval/ocr_eval/scans",
                    help="Katalog z podfolderami level1/2/3.")
    ap.add_argument("--gt_dir", type=str, default="eval/ocr_eval/gt",
                    help="Katalog z plikami doc_{id}.gt.txt.")
    ap.add_argument("--levels", type=str, default="level1,level2,level3",
                    help="Poziomy, rozdzielone przecinkami.")
    ap.add_argument("--engines", type=str, default="tesseract,easyocr",
                    help="Silniki do testu: tesseract,easyocr")
    ap.add_argument("--out_dir", type=str, default="eval/ocr_eval/results",
                    help="Gdzie zapisać CSV-y.")
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="Ścieżka do config.yaml (używany też przez pipeline).")
    ap.add_argument("--wer_mode", choices=["strict","nopunct"], default="nopunct",
                    help="WER: 'strict' z interpunkcją; 'nopunct' bez interpunkcji.")
    ap.add_argument("--cer_keep", choices=["all","alnum"], default="alnum",
                    help="CER: 'all' po wszystkich znakach; 'alnum' tylko [0-9a-z].")
    args = ap.parse_args()

    scans_root = Path(args.scans_root)
    gt_dir     = Path(args.gt_dir)
    out_dir    = Path(args.out_dir)
    levels     = [x.strip() for x in args.levels.split(",") if x.strip()]
    engines    = [x.strip().lower() for x in args.engines.split(",") if x.strip()]
    cfg_path   = Path(args.config)

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_engine_on_levels(
        scans_root=scans_root,
        gt_dir=gt_dir,
        engines=engines,
        levels=levels,
        out_dir=out_dir,
        cfg_path=cfg_path,
        wer_mode=args.wer_mode,
        cer_keep=args.cer_keep,
    )
if __name__ == "__main__":
    main()
