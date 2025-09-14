# Document Classification & OCR Pipeline

## Overview

This project provides an end-to-end machine learning pipeline for automatically classifying technical documents and drawings. Key features include:
- Synthetic data generation (structured PDFs/images)
- Scan simulation with common artifacts
- OCR extraction using Tesseract or EasyOCR
- Classification using machine learning and deep learning (Logistic Regression & CNN)

## Folder Structure

```
project/
├── data/
│   ├── assets/            # Textures, backgrounds for document generation
│   ├── scans/             # Simulated scans (from PDFs)
│   │   └── docs/, technical_drawings/
│   └── processed/
│       ├── ocr/           # Extracted OCR .txt outputs
│       └── metrics/       # Model evaluation metrics
├── models/                # Trained models (LR/CNN), organized by level and OCR engine
├── docpipe/
│   ├── etl/
│   │   └── extras/generators/  # Synthetic document generator scripts
│   ├── ocr/                   # OCR utility functions
│   ├── models/                # Training scripts for LR/CNN models
│   └── cli/
│       ├── pipeline_main.py   # Main classification pipeline
│       ├── classification_utils.py
│       ├── ocr_utils.py
│       └── file_utils.py
├── tests/
│   ├── input/           # Folder monitored by pipeline (drop test files here)
│   └── output/          # Classified files moved here
├── config.yaml          # Central configuration
└── main.py              # CLI entry point (runs pipeline_main)
```

## How It Works

1. Input: Drop an image (.jpg) or PDF (.pdf) into tests/input/.
2. OCR: The file is processed with OCR (EasyOCR or Tesseract).
3. Stage 1: A model determines if it is a document or a technical drawing.
4. Stage 2: If identified as a document, another model classifies its type (BOM, PDS, etc.).
5. Output: The file is renamed and moved to tests/output/.
6. Logging: Classification metadata (filename, model, prediction, time) is logged to an Excel file.

## Pipeline Components

### 1. Synthetic Document Generation
```
python -m docpipe.etl.extras.generators.generate_data
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
python -m docpipe.models.cnn_doc_type_classifier --config config.yaml
python -m docpipe.models.cnn_doc_vs_drw --config config.yaml
```

## Running the Pipeline

To run the main classification pipeline (continuously monitoring the input folder):

```
python main.py
```

It continuously monitors `tests/input/` and classifies every new file.

## Config Explanation (config.yaml)

Key configuration options:
- OCR engine: easy_ocr or tesseract_ocr
- Classification models: cnn or tfidf_lr (Logistic Regression)
- Complexity level: level1, level2, or level3
- Paths: Auto-resolved based on project_root in the config file

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
| level2  | Real-life examples (typical quality)    |
| level3  | Heavily distorted, low-quality scans  |


## Author Notes

- Written and tested with Python 3.10+
- GPU not required, but EasyOCR is faster with CUDA
- Vectorizer `.pkl` files are stored for LR inference
- CNN models use ResNet18 (doc type) and MobileNetV2 (doc vs drw)
