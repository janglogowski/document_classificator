import argparse
import os
import shutil
from pathlib import Path
import yaml
import tempfile
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def import_generators():
    from docpipe.etl.database_gen.generators.main import generate_database
    return generate_database

def import_scan_imitator():
    from docpipe.etl.scan_imitator import imitate_scans
    return imitate_scans

def rm_tree(path: Path) -> None:
    if path.exists():
        for _ in range(3):
            try:
                shutil.rmtree(path, ignore_errors=True)
                if not path.exists():
                    break
            except Exception:
                pass

def generate_and_scan_default(cfg: dict, level: str) -> None:

    """
    DEFAULT:
      1) Generate PDFs into a temp staging dir: <TMP>/_stage_pdf/<level>/<TYPE>
      2) Imitate scans from staging -> data/scans/docs/<level>/<TYPE> (JPG)
      3) Remove staging dir
    """

    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))
    out_scans = ROOT / cfg["paths"]["data"]["scans"]["docs"][level]
    ensure_dir(out_scans)

    stage_pdf = Path(tempfile.mkdtemp(prefix=f"stage_pdf_{level}_"))

    generate_database = import_generators()
    imitate_scans = import_scan_imitator()

    print(f"\n=== [DEFAULT] Generating PDFs into staging: {stage_pdf} ===")
    generate_database(cfg=cfg, mode="default", output_folder=str(stage_pdf), level=None)

    print(f"\n=== [DEFAULT] Scan imitation: {stage_pdf} -> {out_scans} (level={level}) ===")
    imitate_scans(
        mode="default",
        level=level,
        input_folder=str(stage_pdf),
        output_folder=str(out_scans),
    )

    print(f"\n=== [DEFAULT] Removing staging: {stage_pdf} ===")
    rm_tree(stage_pdf)
    print("\n=== [DEFAULT] Done. ===")


def generate_and_scan_test(cfg: dict, level: str) -> None:

    """
    TEST:
      1) Generate PDFs into staging: <TMP>/stage_pdf_test_<random>/<level>  (flat)
      2) Imitate scans (mode='test'), które zapisują JPG w *stagingu*
      3) Przenieś JPG do tests/test_data/test_docs/<level>
      4) Usuń staging
    """

    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))
    test_cfg = cfg["generator_settings"]["test"]

    out_scans = ROOT / test_cfg["output_folder"] / level
    ensure_dir(out_scans)

    stage_root = Path(tempfile.mkdtemp(prefix="stage_pdf_test_"))
    level_stage = stage_root / level
    ensure_dir(level_stage)

    generate_database = import_generators()
    imitate_scans = import_scan_imitator()

    print(f"\n=== [TEST] Generating test PDFs into staging: {level_stage} ===")
    generate_database(cfg=cfg, mode="test", output_folder=str(stage_root), level=level)

    print(f"\n=== [TEST] Scan imitation (mode=test): {level_stage} -> {out_scans} ===")
    imitate_scans(
        mode="test",
        level=level,
        input_folder=str(level_stage),
        output_folder=str(out_scans)
    )

    moved = 0
    for jpg in level_stage.glob("*.jpg"):
        dest = out_scans / jpg.name
        try:
            shutil.move(str(jpg), str(dest))
            moved += 1
        except Exception:
            pass
    print(f"[TEST] Moved {moved} JPG file(s) to: {out_scans}")

    print(f"\n=== [TEST] Removing staging: {stage_root} ===")
    rm_tree(stage_root)
    print("\n=== [TEST] Done. ===")


def generate_and_scan_eval(cfg: dict, levels: list[str]) -> None:
    from pathlib import Path
    import shutil, tempfile
    import pandas as pd

    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))
    eval_root = ROOT / "eval" / "ocr_eval"
    pdf_dir   = eval_root / "pdf"
    gt_dir    = eval_root / "gt"
    scans_dir = eval_root / "scans"

    ensure_dir(eval_root); ensure_dir(pdf_dir); ensure_dir(gt_dir); ensure_dir(scans_dir)

    stage_root = Path(tempfile.mkdtemp(prefix="stage_pdf_eval_"))
    ensure_dir(stage_root)

    generate_database = import_generators()
    print(f"\n=== [EVAL] Generating eval PDFs into staging: {stage_root} ===")
    generate_database(cfg=cfg, mode="eval", output_folder=str(stage_root), level=None)

    try:
        from pdfminer.high_level import extract_text as _pdf_extract_text
    except Exception:
        _pdf_extract_text = None

    def _export_fallback_gt(pdf_path: Path) -> str:
        if _pdf_extract_text is None:
            return ""
        try:
            return _pdf_extract_text(str(pdf_path))
        except Exception:
            return ""

    pdf_paths = sorted(stage_root.rglob("*.pdf"))
    if not pdf_paths:
        print("[EVAL][WARN] No PDFs generated in staging. Nothing to do.")
        rm_tree(stage_root)
        return

    print(f"[EVAL] Found {len(pdf_paths)} PDF(s) in staging. Moving & renaming to doc_{{id}}...")

    manifest_rows = []
    next_id = 1
    for src_pdf in pdf_paths:
        doc_type = src_pdf.parent.name

        stem_new = f"doc_{next_id}"
        dst_pdf = pdf_dir / f"{stem_new}.pdf"
        shutil.move(str(src_pdf), str(dst_pdf))

        src_gt = src_pdf.with_suffix(".gt.txt")
        if src_gt.exists():
            gt_text = src_gt.read_text(encoding="utf-8", errors="ignore")
        else:
            gt_text = _export_fallback_gt(dst_pdf)
        (gt_dir / f"{stem_new}.gt.txt").write_text(gt_text, encoding="utf-8")

        manifest_rows.append({"doc_id": next_id, "doc_type": doc_type})
        next_id += 1

    pd.DataFrame(manifest_rows).to_csv(eval_root / "manifest.csv", index=False)
    print(f"[EVAL] Stored PDFs in: {pdf_dir}")
    print(f"[EVAL] Stored GT   in: {gt_dir}")
    print(f"[EVAL] Saved manifest in: {eval_root / 'manifest.csv'}")

    imitate_scans = import_scan_imitator()
    for lv in levels:
        out_scans = scans_dir / lv
        ensure_dir(out_scans)
        print(f"\n=== [EVAL] Scan imitation: {pdf_dir} -> {out_scans} (level={lv}) ===")
        imitate_scans(
            mode="eval",
            level=lv,
            input_folder=str(pdf_dir),
            output_folder=str(out_scans))

        for img in sorted(out_scans.iterdir()):
            if not img.is_file():
                continue
            if f"_{lv}" in img.stem:
                continue
            new_name = f"{img.stem}_{lv}{img.suffix}"
            img.rename(out_scans / new_name)

    print(f"\n=== [EVAL] Removing staging: {stage_root} ===")
    rm_tree(stage_root)
    print("\n=== [EVAL] Done. ===")


def main():
    parser = argparse.ArgumentParser(description="Generate PDFs, convert to scan-like images, and clean staging.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["default", "test", "eval"], required=True, help="default = full dataset; test = small pack; eval = eval set")
    parser.add_argument("--levels", type=str, default="level1,level2,level3", help="Comma-separated levels, e.g. level1,level2,level3")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    if args.mode == "default":
        for lv in levels:
            generate_and_scan_default(cfg, level=lv)
    elif args.mode == "eval":
        generate_and_scan_eval(cfg, levels=levels)
    else:
        for lv in levels:
            generate_and_scan_test(cfg, level=lv)

if __name__ == "__main__":
    main()
