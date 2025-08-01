import argparse
import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

import yaml
from sklearn.metrics import classification_report


# --------- utils --------- #
def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# --------- data discovery --------- #
def discover_docs(ocr_docs_root: Path) -> List[Tuple[Path, str]]:
    """
    Zwraca listę (plik_txt, doc_family). Oczekiwana struktura:
      <ocr_docs_root>/<doc_family>/*.txt
    """
    items: List[Tuple[Path, str]] = []
    if not ocr_docs_root.exists():
        return items

    for family_dir in sorted(d for d in ocr_docs_root.iterdir() if d.is_dir()):
        family = family_dir.name.lower()
        for p in sorted(family_dir.rglob("*.txt")):
            if p.is_file():
                items.append((p, family))
    return items


# --------- rules --------- #
RULES: Dict[str, List[str]] = {
    "bill_of_materials": [
        "bill of material", "component breakdown", "bom report",
        "unit price", "line total", "rate", "amount", "uom",
        "total amount:", "approved by", "order qty"
    ],
    "product_data_sheet": [
        "product data sheet", "unit configuration", "component overview",
        "design pressure", "flow rate", "motor power", "voltage", "protection class",
        "service interval", "dimensions"
    ],
    "quality_checklist": [
        "quality checklist", "inspection checklist", "qc checklist",
        "sampling level", "classification", "critical", "major", "minor",
        "supervised by", "customer specific", "aql level"
    ],
    "daily_report": [
        "daily production report", "production log", "shift summary",
        "target qty", "actual qty", "scrap qty", "rework qty",
        "machine id", "operation", "operator", "duration (min)",
        "temperature", "energy (kwh)"
    ],
    "inspection_report": [
        "dimensional inspection", "measurement summary", "tolerance check",
        "nominal (mm)", "measured (mm)", "deviation", "status (ok/nok)",
        "pass/fail", "inspection results"
    ],
    "maintenance_log": [
        "maintenance log", "maintenance checklist", "equipment record",
        "maintenance type", "performed by", "downtime (hrs)",
        "last maintenance", "spare parts used", "authorized by"
    ],
}

ALL_CLASSES = [
    "bill_of_materials",
    "product_data_sheet",
    "quality_checklist",
    "daily_report",
    "inspection_report",
    "maintenance_log",
    "other",
]

def rule_classifier(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t)

    for cls, keys in RULES.items():
        for k in keys:
            if k in t:
                return cls
    return "other"


# --------- main --------- #
def main():
    ap = argparse.ArgumentParser(description="Prosty klasyfikator rule-based dla typu dokumentu.")
    ap.add_argument("--config", type=str, default="config.yaml", help="Ścieżka do config.yaml")
    ap.add_argument("--engine", type=str, required=True, choices=["tesseract_ocr", "easy_ocr"], help="OCR engine")
    ap.add_argument("--level", type=str, required=True, choices=["level1", "level2", "level3"], help="Poziom skanów")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    ROOT = Path(os.path.abspath(cfg["paths"]["project_root"]))

    ocr_root = ROOT / cfg["paths"]["data"]["processed"]["ocr_root"]
    docs_root = ocr_root / args.engine / args.level / "docs"

    items = discover_docs(docs_root)
    if not items:
        print(f"[INFO] Brak plików TXT w {docs_root}")
        return

    y_true: List[str] = []
    y_pred: List[str] = []

    for p, family in items:
        text = read_text(p)
        if not text.strip():
            continue

        fam = family.lower()
        alias = {
            "bom": "bill_of_materials",
            "bill_of_materials": "bill_of_materials",
            "quality_checklist": "quality_checklist",
            "daily_report": "daily_report",
            "inspection_report": "inspection_report",
            "maintenance_log": "maintenance_log",
            "product_data_sheet": "product_data_sheet",
        }
        y_t = alias.get(fam, "other")
        y_p = rule_classifier(text)

        y_true.append(y_t)
        y_pred.append(y_p)

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    incorrect = total - correct
    accuracy = (correct / total) if total else 0.0

    class_report = classification_report(
        y_true, y_pred, digits=4, output_dict=True, zero_division=0, labels=ALL_CLASSES
    )
    print("=== Rule-based Classifier ===")
    print(f"Engine/Level : {args.engine} / {args.level}")
    print(f"Docs root    : {docs_root}")
    print(f"Total docs   : {total}")
    print(f"Correct      : {correct}")
    print(f"Incorrect    : {incorrect}")
    print(f"Accuracy     : {accuracy:.4%}\n")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0, labels=ALL_CLASSES))

    out_dir = ROOT / cfg["paths"]["data"]["models"]["rule_based_classifier"] / args.engine
    ensure_dir(out_dir)
    out_path = out_dir / f"rule_based_metrics_{args.level}.json"

    metrics = {
        "engine": args.engine,
        "level": args.level,
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "classes": ALL_CLASSES,
        "classification_report": class_report,
        "docs_root": str(docs_root),
    }

    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved metrics >>> {out_path}")


if __name__ == "__main__":
    main()
