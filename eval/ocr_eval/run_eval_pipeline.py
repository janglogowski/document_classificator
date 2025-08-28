import argparse
import subprocess
from pathlib import Path
import sys
from typing import List

def as_levels(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Generate eval set -> make scans -> run OCR evaluation.")
    ap.add_argument("--levels", type=str, default="level1,level2,level3",
                    help="Comma-separated levels (default: all 3).")
    ap.add_argument("--engines", type=str, default="tesseract,easyocr",
                    help="Engines to test (default: both).")
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="Path to config.yaml.")
    ap.add_argument("--wer_mode", choices=["strict", "nopunct"], default="nopunct",
                    help="WER: 'strict' liczy interpunkcję; 'nopunct' ignoruje interpunkcję.")
    ap.add_argument("--cer_keep", choices=["all", "alnum"], default="alnum",
                    help="CER: 'all' po wszystkich znakach; 'alnum' tylko [0-9a-z].")
    args = ap.parse_args()

    # project root
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))

    # # STEP 1) Generate eval dataset (PDF + GT)
    # print("\n=== STEP 1: Generating evaluation dataset (PDF + GT) ===")
    # gen_cmd = [
    #     "python", str(root / "docpipe" / "etl" / "database_gen" / "generate_data.py"),
    #     "--mode", "eval",
    #     "--levels", args.levels,
    #     "--config", args.config
    # ]
    # subprocess.run(gen_cmd, check=True)

    # # STEP 2) Convert PDFs -> scans per level (explicit call to imitate_scans)
    # print("\n=== STEP 2: Converting PDFs -> scans for requested levels ===")
    # from docpipe.etl.scan_imitator import imitate_scans

    # pdf_dir   = root / "eval" / "ocr_eval" / "pdf"
    # scans_dir = root / "eval" / "ocr_eval" / "scans"
    # scans_dir.mkdir(parents=True, exist_ok=True)

    # levels = as_levels(args.levels)
    # for lv in levels:
    #     out_level = scans_dir / lv
    #     out_level.mkdir(parents=True, exist_ok=True)
    #     print(f"\n--- [SCAN] {lv}: {pdf_dir} -> {out_level} ---")
    #     try:
    #         imitate_scans(
    #             mode="eval",
    #             level=lv,
    #             input_folder=str(pdf_dir),
    #             output_folder=str(out_level)
    #         )
    #     except FileNotFoundError as e:
    #         print(f"[WARN] Scan imitation failed due to missing asset ({e}). "
    #               f"Ustaw w config.yaml: augmentation.levels.{lv}.process_image.paper_texture_probability: 0.0 "
    #               "albo dodaj brakujące tekstury (data/assets/textures/*).")
    #         raise

    #     changed = 0
    #     for img in out_level.iterdir():
    #         if not img.is_file():
    #             continue
    #         stem, suf = img.stem, img.suffix
    #         if f"_{lv}" not in stem:
    #             img.rename(out_level / f"{stem}_{lv}{suf}")
    #             changed += 1
    #     if changed:
    #         print(f"[SCAN] Renamed {changed} file(s) to include _{lv} suffix.")

    # total_png = 0
    # for lv in levels:
    #     cnt = sum(1 for p in (scans_dir / lv).glob("*.png"))
    #     total_png += cnt
    #     print(f"[INFO] scans/{lv}: {cnt} PNG")
    # print(f"[INFO] Total scans: {total_png} PNG")

    # STEP 3) Run OCR evaluation
    print("\n=== STEP 3: Running OCR evaluation ===")
    eval_cmd = [
        "python", str(root / "eval" / "ocr_eval" / "evaluate_ocr.py"),
        "--levels", args.levels,
        "--engines", args.engines,
        "--config", args.config,
        "--wer_mode", args.wer_mode,
        "--cer_keep", args.cer_keep]
    
    subprocess.run(eval_cmd, check=True)

    print("\n=== DONE: Results available in eval/ocr_eval/results/ ===")

if __name__ == "__main__":
    main()
