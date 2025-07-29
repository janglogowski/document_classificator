import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

from .BOM import generate_bom_document
from .Quality_Chceklist import generate_quality_checklist
from .Daily_Production_Report import generate_daily_production_report
from .Dimensional_Inspection_Report import generate_dimensional_inspection_report
from .Maintenance_log import generate_maintenance_log
from .Product_Data_Sheet import generate_product_data_sheet

def generate_database(mode: str = "default", output_folder: str = None, level: str = None):

    """
    Generate a set of PDF documents.
    mode:
      - "default": full training set, writes into subfolders per-type.
      - "test":    small test set, writes all into one level folder (no subfolders).
    """
    
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Nie znaleziono config.yaml w {config_path}")
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, cfg["paths"]["project_root"]))
    if not os.path.exists(ROOT):
        raise FileNotFoundError(f"Data root nie istnieje: {ROOT}")

    gencfg = cfg["generator_settings"][mode]
    TYPES = cfg["paths"]["data"]["generators"]

    base = output_folder or os.path.join(ROOT, gencfg["output_folder"])
    os.makedirs(base, exist_ok=True)

    N = gencfg["num_documents"]
    print(f"=== Starting '{mode}' generation: {N} of each type into '{base}' ===")

    for i in range(1, N + 1):
        for key, gen_func in [
            ("bom", generate_bom_document),
            ("quality_checklist", generate_quality_checklist),
            ("daily_report", generate_daily_production_report),
            ("inspection_report", generate_dimensional_inspection_report),
            ("maintenance_log", generate_maintenance_log),
            ("product_data_sheet", generate_product_data_sheet),
        ]:
            if mode == "test":
                if level is None:
                    raise ValueError("Level must be specified for test mode")
                dest = os.path.join(base, level)
            else:
                dest = os.path.join(base, TYPES[key])

            os.makedirs(dest, exist_ok=True)
            print(f"[{i}/{N}] Generating {key} → {dest}")
            gen_func(i, dest)

    print(f"=== Finished generating into '{base}' ===")

if __name__ == "__main__":
    generate_database("default")
