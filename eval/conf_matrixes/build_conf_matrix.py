import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

EXCEL_PATH = r"C:\Users\janek\Documents\GitHub\document_classificator\eval\conf_matrixes\CNN_ONLY.xlsx"
OUT_DIR    = Path(r"C:\Users\janek\Documents\GitHub\document_classificator\eval\conf_matrixes")
NORMALIZE  = True
# ============================

CLASS_MAP = {
    r"^bom": "BOM",
    r"^daily[_\-]?report": "DAILY_REPORT",
    r"^dim[_\-]?inspection[_\-]?report": "INSPECTION_REPORT",
    r"^maintenance[_\-]?log": "MAINTENANCE_LOG",
    r"^product[_\-]?data[_\-]?sheet": "PRODUCT_DATA_SHEET",
    r"^quality[_\-]?checklist": "QUALITY_CHECKLIST",
    r"^tech[_\-]?drw": "TECH_DRW",
}

ALL_LABELS = [
    "BOM",
    "DAILY_REPORT",
    "INSPECTION_REPORT",
    "MAINTENANCE_LOG",
    "PRODUCT_DATA_SHEET",
    "QUALITY_CHECKLIST",
    "TECH_DRW",
]

def infer_true_type(fname: str) -> str:
    base = str(fname).split("/")[-1].split("\\")[-1].lower()
    for pat, lab in CLASS_MAP.items():
        if re.match(pat, base):
            return lab
    return "UNKNOWN"

def normalize_rows(cm: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return np.nan_to_num(cmn)

def plot_cm(cm, labels, title, outpath, normalize=True):
    arr = normalize_rows(cm) if normalize else cm
    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, interpolation="nearest", cmap="RdYlGn", vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = f"{arr[i, j]:.2f}" if normalize else f"{int(arr[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)

def engine_tag(eng: str) -> str:
    e = (eng or "").strip().lower()
    if e in {"-", "none", "cnn"}:
        return "CNN"
    if e in {"tesseract", "tesseract_ocr"}:
        return "tesseract_ocr"
    if e in {"easyocr", "easy_ocr"}:
        return "easy_ocr"
    return eng  

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_PATH)

    required = {"input_file", "final_pred", "ocr_engine", "train_level", "test_level"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}")

    for c in ["input_file","final_pred","ocr_engine","train_level","test_level"]:
        df[c] = df[c].astype(str).str.strip()

    df["true_type"] = df["input_file"].apply(infer_true_type)
    df = df[df["true_type"] != "UNKNOWN"].copy()

    df["ocr_engine_tag"] = df["ocr_engine"].apply(engine_tag)
    df["train_level"] = df["train_level"].replace({"": "-", "nan": "-", "None": "-"})
    df["test_level"]  = df["test_level"].replace({"": "-", "nan": "-", "None": "-"})

    for eng in sorted(df["ocr_engine_tag"].unique()):
        sub = df[df["ocr_engine_tag"] == eng]
        if sub.empty:
            continue
        cm = confusion_matrix(sub["true_type"], sub["final_pred"], labels=ALL_LABELS)
        plot_cm(cm, ALL_LABELS, f"Final CM — {eng} (ALL LEVELS)", OUT_DIR / f"cm_final_{eng}_ALL.png", NORMALIZE)

    print(f"Saved in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
