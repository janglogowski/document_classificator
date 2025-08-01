from pathlib import Path
from typing import Optional, Dict, Callable, List, Tuple

# Local generators 
from .bom import generate_bom_document
from .quality_checklist import generate_quality_checklist
from .daily_report import generate_daily_production_report
from .inspection_report import generate_dimensional_inspection_report
from .maintenance_log import generate_maintenance_log
from .product_data_sheet import generate_product_data_sheet


def _resolve_base_dir(cfg: dict, mode: str, output_folder: Optional[str]) -> Path:
    """
    Resolve the base output directory for generated PDFs.

    - If `output_folder` is provided, use it as-is.
    - Otherwise fall back to cfg['generator_settings'][mode]['output_folder'] under project_root.
    """
    project_root = Path(cfg["paths"]["project_root"]).resolve()
    if output_folder:
        return Path(output_folder).resolve()
    return (project_root / cfg["generator_settings"][mode]["output_folder"]).resolve()


def generate_database(
    cfg: dict,
    mode: str = "default",
    output_folder: Optional[str] = None,
    level: Optional[str] = None) -> None:
    """
    Generate a set of synthetic PDFs using document generators.

    Modes:
      - "default": full training set. Writes into subfolders per type.
      - "test": small test pack. Writes all PDFs into a single flat folder.
    """
    if mode not in ("default", "test"):
        raise ValueError("mode must be 'default' or 'test'")

    # Map generator keys -> functions
    generators: List[Tuple[str, Callable[[int, str], None]]] = [
        ("bom", generate_bom_document),
        ("quality_checklist", generate_quality_checklist),
        ("daily_report", generate_daily_production_report),
        ("inspection_report", generate_dimensional_inspection_report),
        ("maintenance_log", generate_maintenance_log),
        ("product_data_sheet", generate_product_data_sheet),
    ]

    # Per-type folder names taken from config
    types_map: Dict[str, str] = cfg["paths"]["data"]["generators"]

    # How many documents per type
    num_docs: int = int(cfg["generator_settings"][mode]["num_documents"])

    # Base directory for generated PDFs
    base_dir = _resolve_base_dir(cfg, mode, output_folder)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Decide final output directory layout
    if mode == "test":
        if not level:
            raise ValueError("Level must be specified for test mode.")
        out_dir = base_dir / level          # flat folder for all types
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== Generating {num_docs} PDFs per type into: {out_dir} (mode=test) ===")
    else:
        # In default mode we expect per-type subfolders under base_dir
        out_dir = base_dir
        print(f"=== Generating {num_docs} PDFs per type into: {out_dir} (mode=default) ===")

    # Main generation loop
    for i in range(1, num_docs + 1):
        for key, gen_func in generators:
            if mode == "test":
                dest = out_dir                              # flat: same folder for all
            else:
                type_subdir = types_map[key]
                dest = out_dir / type_subdir

            dest.mkdir(parents=True, exist_ok=True)
            print(f"[{i}/{num_docs}] Generating {key} -> {dest}")
            gen_func(i, str(dest))

    print("=== Generation finished ===")


if __name__ == "__main__":
    pass
