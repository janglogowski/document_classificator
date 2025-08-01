from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import numpy as np, cv2
from pdf2image import convert_from_path

class EasyOCREngine:
    IMG_EXTS: Tuple[str, ...]
    PDF_EXT: str

    def __init__(self, cfg: dict, poppler_path: str):
        import easyocr
        self.easyocr = easyocr

        ecfg = cfg.get("ocr", {}).get("easy_ocr", cfg["easy_ocr"])
        self.IMG_EXTS = tuple(ecfg["image_extensions"])
        self.PDF_EXT  = ecfg["pdf_extension"]
        self.PDF_DPI  = int(ecfg["pdf_dpi"])
        self.MAX_DIM  = int(ecfg["max_dim"])
        self.poppler_path = poppler_path

        raw_lang = ecfg["lang"]
        self.LANGS = [raw_lang] if isinstance(raw_lang, str) else list(raw_lang)
        self.reader = self.easyocr.Reader(self.LANGS, gpu=False)

    def _resize_max_dim(self, img, max_dim: int):
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / float(max(h, w))
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    def _ocr_image_array(self, arr_bgr) -> Tuple[str, List[float]]:
        arr_bgr = self._resize_max_dim(arr_bgr, self.MAX_DIM)
        arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
        raw = self.reader.readtext(arr_rgb, detail=1)
        texts = [t[1] for t in raw]
        confs = [float(t[2]) for t in raw]
        return "\n".join(texts), confs

    def ocr_path(self, path: Path) -> Tuple[str, List[float]]:
        ext = path.suffix.lower()
        if ext in self.IMG_EXTS:
            img = cv2.imread(str(path))
            if img is None:
                return "", []
            return self._ocr_image_array(img)

        if ext == self.PDF_EXT:
            pages = convert_from_path(str(path), dpi=self.PDF_DPI, poppler_path=self.poppler_path)
            chunks, confs = [], []
            for pg in pages:
                arr = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
                t, c = self._ocr_image_array(arr)
                chunks.append(t); confs.extend(c)
            return "\n\n".join(chunks), confs

        return "", []
