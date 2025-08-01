import os
import time
import shutil
from datetime import datetime

import yaml
from docpipe.cli.ocr_utils import get_ocr_function
from docpipe.cli.classification_utils import (load_vsd_model, load_doc_type_model,predict_doc_vs_drw, predict_doc_type)
from docpipe.cli.file_utils import generate_filename, log_metadata

# === Config === #
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
ROOT = os.path.abspath(cfg["paths"]["project_root"])

# === User parameters === #
ocr_engine     = "easy_ocr"         # "easy_ocr"/"tesseract_ocr"
vsd_classifier = "cnn"              # "cnn"/"tfidf_lr"
doc_classifier = "tfidf_lr"         # "cnn"/"tfidf_lr"
level          = "level3"

# === Paths === #
P = cfg["paths"]
INPUT_FOLDER  = os.path.join(ROOT, P["tests"]["input"])
OUTPUT_FOLDER = os.path.join(ROOT, P["tests"]["output"])
METADATA_CSV  = os.path.join(ROOT, P["tests"]["metadata"])
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Initialize OCR === #
ocr_image = get_ocr_function(ocr_engine, cfg)

# === Load models === #
vsd_model = load_vsd_model(vsd_classifier, level, cfg)
doc_model, doc_names = load_doc_type_model(doc_classifier, ocr_engine, level, cfg)

# === Watch loop === #
print(f"\n=== Watching {INPUT_FOLDER} | OCR={ocr_engine} | VSD={vsd_classifier} | DT={doc_classifier} ===\n")

while True:
    for fn in sorted(os.listdir(INPUT_FOLDER)):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in cfg[ocr_engine]["image_extensions"] + [cfg[ocr_engine]["pdf_extension"]]:
            continue

        src = os.path.join(INPUT_FOLDER, fn)
        print(f"[→] {fn}")
        start_time = time.time()

        text = ""
        if vsd_classifier == "tfidf_lr" or doc_classifier == "tfidf_lr":
            text = ocr_image(src).strip()

        # Stage 1 – doc vs drw
        pred_vsd = predict_doc_vs_drw(src, vsd_classifier, vsd_model, text, cfg)
        if pred_vsd != "document":
            dstype = "tech_drw"
        else:
            # Stage 2 – doc type classification
            dstype = predict_doc_type(src, doc_classifier, (doc_model, doc_names), ocr_image, text, cfg)

        classification_time = round(time.time() - start_time, 3)
        new_name = generate_filename(dstype, ext)
        shutil.move(src, os.path.join(OUTPUT_FOLDER, new_name))
        log_metadata(METADATA_CSV, dstype, src, new_name, ocr_engine, vsd_classifier, doc_classifier, level, classification_time)

        print(f"[OK] → {new_name} (classified in {classification_time}s)\n")

    time.sleep(1.0)
