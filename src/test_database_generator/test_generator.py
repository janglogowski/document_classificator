import os
import sys
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from doc_generators.main_generator import generate_database
from augmentation.scan_imitation_transformer import imitate_scans


def main():
    level = "level3"  # ← zmień na "level2" lub "level3" jeśli chcesz

    # Wczytanie config.yaml
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, cfg["paths"]["project_root"]))
    test_cfg = cfg["generator_settings"]["test"]
    N = test_cfg["num_documents"]

    # Folder wyjściowy: .../test_docs/level1
    BASE_OUT_DIR = os.path.join(ROOT, test_cfg["output_folder"])
    LEVEL_FOLDER = os.path.join(BASE_OUT_DIR, level)
    os.makedirs(LEVEL_FOLDER, exist_ok=True)

    # === Krok 1: Generowanie dokumentów PDF ===
    print(f"\n=== GENERATING {N} OF EACH TEST DOCUMENT TYPE INTO:\n    {LEVEL_FOLDER} ===")
    generate_database(mode="test", output_folder=BASE_OUT_DIR, level=level)

    # === Krok 2: Imitacja skanowania ===
    print(f"\n=== RUNNING SCAN IMITATION ({level}) ON ALL PDFs IN:\n    {LEVEL_FOLDER} ===")
    imitate_scans(mode="test", level=level, input_folder=LEVEL_FOLDER, output_folder=LEVEL_FOLDER)

    # === Krok 3: Usuwanie oryginalnych PDF-ów ===
    print("\n=== CLEANING UP ORIGINAL PDFs ===")
    count = 0
    for f in os.listdir(LEVEL_FOLDER):
        if f.lower().endswith(".pdf"):
            os.remove(os.path.join(LEVEL_FOLDER, f))
            count += 1
    print(f"  Removed {count} PDF file(s)")

    print("\n=== ALL TEST DATA READY ===")


if __name__ == "__main__":
    main()
