import os
import yaml
import time
import json
import numpy as np
from pdf2image import convert_from_path
import easyocr

cfg           = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT          = os.path.abspath(cfg["paths"]["project_root"])
POPPLER_PATH  = cfg["paths"]["poppler"]["bin_path"]

engine = "easy_ocr"
level = "level1"

SCANS_DOCS     = os.path.join(ROOT, cfg["paths"]["data"]["scans"]["docs"][level])
SCANS_DRAWINGS = os.path.join(ROOT, cfg["paths"]["data"]["scans"]["technical_drawings"])
DOCS_OUT       = os.path.join(ROOT, cfg["paths"]["data"]["extracted"][engine][level])
DRAWINGS_OUT   = os.path.join(ROOT, cfg["paths"]["data"]["extracted"][engine]["technical_drawings"] )
OCR_STATS      = os.path.join(ROOT, cfg["paths"]["data"]["ocr_results"])

os.makedirs(DOCS_OUT, exist_ok=True)
os.makedirs(DRAWINGS_OUT, exist_ok=True)
os.makedirs(OCR_STATS, exist_ok=True)

ocr_cfg   = cfg["easy_ocr"]
raw_lang  = ocr_cfg["lang"]
PDF_DPI   = ocr_cfg["pdf_dpi"]
IMG_EXTS  = tuple(ocr_cfg["image_extensions"])
PDF_EXT   = ocr_cfg["pdf_extension"]
MAX_DIM   = ocr_cfg["max_dim"]

if isinstance(raw_lang, str):
    OCR_LANGS = [raw_lang]
elif isinstance(raw_lang, (list, tuple)):
    OCR_LANGS = list(raw_lang)
else:
    raise ValueError("cfg['easy_ocr']['lang'] must be string or list of strings")

reader = easyocr.Reader(OCR_LANGS, gpu=False)

_file_durations_ms = []
_total_words = 0

def ocr_image(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return "", []
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    raw = reader.readtext(img, detail=1)
    texts = [t[1] for t in raw]
    confs = [t[2] for t in raw]
    return "\n".join(texts), confs

def ocr_pdf(path):
    text_blocks = []
    all_confs   = []
    pages = convert_from_path(path, dpi=PDF_DPI, poppler_path=POPPLER_PATH)
    for page in pages:
        arr = np.array(page)[:, :, ::-1]
        raw = reader.readtext(arr, detail=1)
        texts = [t[1] for t in raw]
        confs = [t[2] for t in raw]
        text_blocks.append("\n".join(texts))
        all_confs.extend(confs)
    return "\n\n".join(text_blocks), all_confs

def ocr_file_with_stats(path, base_input):
    global _total_words
    start = time.time()
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXTS:
        text, confs = ocr_image(path)
    elif ext == PDF_EXT:
        text, confs = ocr_pdf(path)
    else:
        return ""
    duration_ms = (time.time() - start) * 1000.0
    _file_durations_ms.append(duration_ms)
    _total_words += len(text.split())
    return text

def batch_ocr(input_root, output_root):
    for dirpath, _, files in os.walk(input_root):
        rel_dir = os.path.relpath(dirpath, input_root)
        tgt_dir = os.path.join(output_root, rel_dir)
        os.makedirs(tgt_dir, exist_ok=True)
        for fname in files:
            if not fname.lower().endswith(IMG_EXTS + (PDF_EXT,)):
                continue
            src = os.path.join(dirpath, fname)
            dst = os.path.join(tgt_dir, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(dst):
                continue
            txt = ocr_file_with_stats(src, input_root)
            with open(dst, "w", encoding="utf-8") as fw:
                fw.write(txt)
            print(f"OCR >>>> {dst}")

def summarize_folder(txt_root, engine, level):
    counts = []
    for dirpath, _, files in os.walk(txt_root):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue
            text = open(os.path.join(dirpath, fname), encoding="utf-8").read().split()
            counts.append(len(text))
    if not counts:
        return None

    arr = np.array(counts)
    summary = {
        "engine": engine,
        "level": level,
        "folder": txt_root,
        "total_files": int(len(arr)),
        "min_words": int(arr.min()),
        "1st_percentile": int(np.percentile(arr, 1)),
        "median_words": int(np.median(arr)),
        "mean_words": float(arr.mean()),
        "99th_percentile": int(np.percentile(arr, 99)),
        "max_words": int(arr.max())
    }

    print(f"\nOCR stats for {txt_root}:")
    for k, v in summary.items():
        if k in ("engine","level","folder"): 
            continue
        print(f"  {k.replace('_',' '):>15}: {v}")
    return summary

if __name__ == "__main__":
    total_start = time.time()
    print("==== STARTING OCR ====")

    batch_ocr(SCANS_DOCS,     DOCS_OUT)
    batch_ocr(SCANS_DRAWINGS, DRAWINGS_OUT)

    print("==== OCR COMPLETE ====")

    total_time_sec = time.time() - total_start
    total_files    = len(_file_durations_ms)
    avg_time_ms    = (sum(_file_durations_ms) / total_files) if total_files else 0.0

    docs_summary  = summarize_folder(DOCS_OUT,     engine, level)
    draw_summary  = summarize_folder(DRAWINGS_OUT, engine, level)

    report = {
        "global": {
            "engine": engine,
            "level": level,
            "total_files": total_files,
            "total_time_sec": round(total_time_sec, 3),
            "avg_time_ms_per_file": round(avg_time_ms, 2)
        },
        "documents": docs_summary,
        "drawings":  draw_summary}

    stats_path = os.path.join(OCR_STATS, f"ocr_stats_{engine}_{level}.json")
    with open(stats_path, "w", encoding="utf-8") as jf:
        json.dump(report, jf, ensure_ascii=False, indent=2)

    print("\n=== GLOBAL OCR TIME ===")
    print(f"  total_files          : {total_files}")
    print(f"  total_time_sec       : {report['global']['total_time_sec']}")
    print(f"  avg_time_ms_per_file : {report['global']['avg_time_ms_per_file']}")
    print(f"\nSaved OCR metrics >>>> {stats_path}")