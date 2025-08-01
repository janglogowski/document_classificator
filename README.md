# Document Classification & OCR Pipeline

## Overview

This project provides an end-to-end machine learning pipeline for automatic classification of technical documents and drawings, including:

- Synthetic data generation (structured PDFs/images)
- Scan simulation with artifacts
- OCR extraction (Tesseract or EasyOCR)
- Machine Learning & Deep Learning classification (Logistic Regression & CNN)
- Organized CLI usage and modular architecture

## Folder Structure

```
project/
├── data/
│   ├── assets/               # Textures, backgrounds for document generation
│   ├── scans/                # Simulated scan images (from PDFs)
│   │   └── docs/, technical_drawings/
│   └── processed/
│       ├── ocr/              # Extracted OCR .txt outputs
│       └── metrics/          # Model evaluation metrics
│
├── models/                   # Trained models (LR/CNN), per level and OCR
├── docpipe/
│   ├── etl/
│   │   └── extras/generators/   # Synthetic document generator
│   ├── ocr/                     # OCR utility functions
│   ├── models/                  # Training scripts for LR/CNN
│   ├── cli/
│   │   ├── pipeline_main.py     # Main classification pipeline
│   │   ├── classification_utils.py
│   │   ├── ocr_utils.py
│   │   └── file_utils.py
│
├── tests/
│   ├── input/               # Folder watched by pipeline (drop test files here)
│   └── output/              # Classified files moved here
│
├── config.yaml              # Central configuration
└── main.py                  # CLI entrypoint (runs pipeline_main)
```

## How It Works

1. Drop a `.jpg` or `.pdf` into `tests/input/`
2. OCR is applied (EasyOCR or Tesseract)
3. Stage 1: model distinguishes between document and technical drawing
4. Stage 2: if document, classify the type (BOM, PDS, etc.)
5. The file is renamed and moved to `tests/output/`
6. Classification metadata is logged to Excel file

## Pipeline Components

### 1. Synthetic Document Generation
```
python docpipe/etl/extras/generators/generate_data.py
```

### 2. OCR Batch Runner
```
python -m docpipe.etl.ocr_batch --engine easy_ocr --level level3 --config config.yaml
```

### 3. Train Logistic Regression Models
```
python -m docpipe.models.lr_doc_type_classifier --engine tesseract_ocr --level level2 --config config.yaml
python -m docpipe.models.lr_doc_vs_drw --engine easy_ocr --level level3 --config config.yaml
```

### 4. Train CNN Classifiers
```
python -m docpipe.models.cnn_doc_type_classifier
python -m docpipe.models.cnn_doc_vs_drw
```

## Run Main Pipeline via CLI

```
python main.py
```

It continuously monitors `tests/input/` and classifies every new file.

## Config Explanation (config.yaml)

- Choose OCR engine: `easy_ocr` or `tesseract_ocr`
- Choose classification models: `cnn` or `tfidf_lr`
- Set complexity level: `level1`, `level2`, `level3`
- Paths are auto-resolved based on `project_root`

## Classification Labels

Document types:

- BOM
- DAILY_REPORT
- INSPECTION_REPORT
- MAINTENANCE_LOG
- PRODUCT_DATA_SHEET
- QUALITY_CHECKLIST

Top-level label:
- document or tech_drw (technical drawing)

## Metadata Logging

Each processed file logs metadata (filename, model types, prediction, time) to:
```
tests/metadata.xlsx
```
Correct predictions (when ground truth is inferred from filename) are highlighted in green.

## Levels

Support for layout complexity levels:

| Level   | Description                  |
|---------|------------------------------|
| level1  | Clean synthetic structure     |
| level2  | Moderate noise/variation      |
| level3  | Realistic layout + artifacts  |

## FAQ

Q: Do I need to retrain CNN every time?  
A: No. Trained models are stored in `models/`, and loaded at runtime.

Q: Why do I need OCR if CNN uses images?  
A: Only `tfidf_lr` models require `.txt` input. CNNs work directly on image data.

Q: Why are models split into doc_type and doc_vs_drw?  
A: This allows the system to first determine if a file is a technical drawing or document, then classify document type only if needed.

## Author Notes

- Written and tested with Python 3.10+
- GPU not required, but EasyOCR is faster with CUDA
- Vectorizer `.pkl` files are stored for LR inference
- CNN models use ResNet18 (doc type) and MobileNetV2 (doc vs drw)