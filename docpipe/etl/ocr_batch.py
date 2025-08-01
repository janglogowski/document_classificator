import argparse
import os
import time
import json
from pathlib import Path
from typing import List, Tuple
import yaml


# ---------------------------- helpers ----------------------------

def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_files(root: Path, img_exts: Tuple[str, ...], pdf_ext: str) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in img_exts or ext == pdf_ext:
                out.append(p)
    out.sort()
    return out


def write_txt(dst_path: Path, text: str) -> None:
    ensure_dir(dst_path.parent)
    dst_path.write_text(text, encoding="utf-8", errors="replace")


def map_scans_to_output(
    scans_root: Path,            
    ocr_root: Path,              
    engine: str,
    level: str,
    is_docs: bool,
    src: Path) -> Path:
    """
    For docs:
      .../processed/ocr/<engine>/<level>/docs/<CLASS>/<file>.txt
    For technical drawings (global, no level):
      .../processed/ocr/<engine>/technical_drawings/<file>.txt  (preserve subfolders if any)
    """
    rel = src.relative_to(scans_root)
    if is_docs:
        doc_class = rel.parts[0] if len(rel.parts) > 1 else "UNKNOWN"
        return ocr_root / engine / level / "docs" / doc_class / (src.stem + ".txt")
    else:
        sub = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path()
        return ocr_root / engine / "technical_drawings" / sub / (src.stem + ".txt")


# ---------------------------- engines ----------------------------

def make_engine(engine_name: str, cfg: dict, poppler_path: str, tess_psm: int, tess_oem: int):
    name = engine_name.lower()
    if name in ("tesseract_ocr", "tesseract"):
        from docpipe.ocr.tesseract_engine import TesseractOCREngine
        return TesseractOCREngine(cfg=cfg, poppler_path=poppler_path, psm=tess_psm, oem=tess_oem)
    elif name in ("easy_ocr", "easyocr"):
        from docpipe.ocr.easyocr_engine import EasyOCREngine
        return EasyOCREngine(cfg=cfg, poppler_path=poppler_path)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


# ---------------------------- runner ----------------------------

def run_ocr_batch(
    config_path: Path,
    engine_name: str,
    level: str,
    tess_psm: int,
    tess_oem: int) -> None:
    """
    Batch OCR:
      - Reads scans from data/scans/docs/<level> and data/scans/technical_drawings
      - Writes text files to data/processed/ocr/<engine>/<level>/docs/... and <engine>/technical_drawings/...
    """
    cfg = load_cfg(config_path)

    poppler_path = (
        cfg.get("paths", {}).get("poppler", {}).get("bin_path")
        or cfg.get("poppler", {}).get("bin_path")
    )

    root = Path(os.path.abspath(cfg["paths"]["project_root"]))

    scans_docs_root = root / cfg["paths"]["data"]["scans"]["docs"][level]
    scans_drw_root  = root / cfg["paths"]["data"]["scans"]["technical_drawings"]

    processed_roots = cfg["paths"]["data"]["processed"]
    ocr_root      = root / processed_roots["ocr_root"]

    # Build OCR engine
    eng = make_engine(engine_name, cfg=cfg, poppler_path=poppler_path, tess_psm=tess_psm, tess_oem=tess_oem)
    img_exts: Tuple[str, ...] = tuple(getattr(eng, "IMG_EXTS"))
    pdf_ext: str = getattr(eng, "PDF_EXT")

    # Collect files
    docs_files = list_files(scans_docs_root, img_exts, pdf_ext)
    drw_files  = list_files(scans_drw_root,  img_exts, pdf_ext)

    # Run OCR
    run_durations_ms: List[float] = []
    total_words = 0
    total_files = 0

    t0 = time.time()

    def process_one(src: Path, is_docs: bool):
        nonlocal total_words, total_files
        dst = map_scans_to_output(
            scans_docs_root if is_docs else scans_drw_root,
            ocr_root, engine_name, level, is_docs, src
        )
        if dst.exists():
            return

        start = time.time()
        text, confs = eng.ocr_path(src)
        elapsed_ms = (time.time() - start) * 1000.0

        write_txt(dst, text)

        run_durations_ms.append(elapsed_ms)
        total_words += len(text.split())
        total_files += 1
        print(f"OCR >>> {dst}")

    for f in docs_files:
        process_one(f, is_docs=True)
    for f in drw_files:
        process_one(f, is_docs=False)

    dt = time.time() - t0
    print(f"\n[OK] OCR finished: {total_files} files in {dt:.2f}s → {ocr_root}")

    # Save one aggregated metrics JSON
    metrics_root = root / processed_roots["metrics_root"] / "ocr"
    ensure_dir(metrics_root)
    avg_ms = float(sum(run_durations_ms) / total_files) if total_files else 0.0

    report = {
        "engine": engine_name,
        "level": level,
        "total_files": total_files,
        "total_time_sec": round(dt, 3),
        "avg_time_ms_per_file": round(avg_ms, 2),
        "total_words": total_words,
        "sources": {
            "docs_root": str(scans_docs_root),
            "technical_drawings_root": str(scans_drw_root),
        },
        "dest_root": str(ocr_root),
    }
    stats_name = f"ocr_stats_{engine_name}_{level}.json"
    (metrics_root / stats_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved OCR metrics >>> {metrics_root / stats_name}")


def main():
    ap = argparse.ArgumentParser(description="Batch OCR: scans → processed/ocr (no per-file meta).")
    ap.add_argument("--config",   type=str, default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--engine",   type=str, choices=["tesseract_ocr", "easy_ocr"], required=True,
                    help="OCR engine to use")
    ap.add_argument("--level",    type=str, choices=["level1", "level2", "level3"], required=True,
                    help="Scan level to process (for docs/).")
    ap.add_argument("--tess-psm", type=int, default=6, help="Tesseract PSM (layout), typical 6 or 11.")
    ap.add_argument("--tess-oem", type=int, default=1, help="Tesseract OEM (LSTM=1).")
    args = ap.parse_args()

    run_ocr_batch(
        config_path=Path(args.config),
        engine_name=args.engine,
        level=args.level,
        tess_psm=args.tess_psm,
        tess_oem=args.tess_oem,
    )


if __name__ == "__main__":
    main()
