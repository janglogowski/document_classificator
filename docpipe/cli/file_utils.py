import os
import uuid
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill

def generate_filename(doc_type, ext):
    uid = uuid.uuid4().hex[:6]
    safe = doc_type.replace(" ", "_").lower()
    return f"{safe}_{uid}{ext}"

def log_metadata(excel_path, doc_type, src, new_name, ocr, vsd, doc_cls, level, duration):
    excel_path = excel_path.replace(".csv", ".xlsx")
    file_exists = os.path.exists(excel_path)

    headers = [
        "input_file", "ocr_engine", "vsd_classifier", "doc_classifier", "level",
        "predicted_type", "internal_name", "timestamp", "classification_time_sec"
    ]
    row = [
        os.path.basename(src), ocr, vsd, doc_cls, level,
        doc_type, new_name, datetime.now().isoformat(), duration
    ]

    if file_exists:
        wb = load_workbook(excel_path)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "metadata"
        ws.append(headers)

    ws.append(row)

    predicted = doc_type.lower()
    filename = os.path.basename(src).split('.')[0].lower()
    if predicted in filename:
        green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        last = ws.max_row
        for col in range(1, len(headers)+1):
            ws.cell(last, col).fill = green

    wb.save(excel_path)
