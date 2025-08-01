# generate_data.py
import argparse
import os
import shutil
import tempfile
from pathlib import Path
import yaml

def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def import_generators():
    from docpipe.etl.extras.generators.main import generate_database
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
    import tempfile

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
    import tempfile
    import shutil
    from pathlib import Path

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



def main():
    parser = argparse.ArgumentParser(description="Generate PDFs, convert to scan-like JPGs, and remove staging.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["default", "test"], required=True, help="default = full dataset; test = small pack")
    parser.add_argument("--levels", type=str, default="level3", help="Comma-separated levels, e.g. level1,level2,level3")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]

    if args.mode == "default":
        for lv in levels:
            generate_and_scan_default(cfg, level=lv)
    else:
        for lv in levels:
            generate_and_scan_test(cfg, level=lv)

if __name__ == "__main__":
    main()
