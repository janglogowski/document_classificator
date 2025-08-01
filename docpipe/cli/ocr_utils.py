import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

def get_ocr_function(engine, cfg):
    img_exts = tuple(cfg[engine]["image_extensions"])
    pdf_ext = cfg[engine]["pdf_extension"]
    max_dim = cfg[engine]["max_dim"]
    lang    = cfg[engine]["lang"]
    poppler = cfg["paths"]["poppler"]["bin_path"]

    if engine == "easy_ocr":
        import easyocr
        reader = easyocr.Reader([lang] if isinstance(lang, str) else lang, gpu=False)
        def ocr_easyocr(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in img_exts:
                img = cv2.imread(path)
                if img is None: return ""
                h, w = img.shape[:2]
                if max(h, w) > max_dim:
                    s = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w*s), int(h*s)))
                return "\n".join(reader.readtext(img, detail=0))
            elif ext == pdf_ext:
                pages = convert_from_path(path, dpi=cfg[engine]["pdf_dpi"], poppler_path=poppler)
                out = ["\n".join(reader.readtext(np.array(pg)[:, :, ::-1], detail=0)) for pg in pages]
                return "\n\n".join(out)
            return ""
        return ocr_easyocr

    elif engine == "tesseract_ocr":
        pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_ocr"]["tesseract_cmd"]
        def ocr_tesseract(path):
            ext = os.path.splitext(path)[1].lower()
            text = ""
            if ext in img_exts:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: return ""
                h, w = img.shape
                if max(h, w) > max_dim:
                    s = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w*s), int(h*s)))
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(img, lang=lang)
            elif ext == pdf_ext:
                pages = convert_from_path(path, dpi=cfg[engine]["pdf_dpi"], poppler_path=poppler)
                for pg in pages:
                    arr = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text += pytesseract.image_to_string(gray, lang=lang) + "\n"
            return text.strip()
        return ocr_tesseract
