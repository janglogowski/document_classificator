from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from pdf2image import convert_from_path


class TesseractOCREngine:
    def __init__(self, cfg: dict, poppler_path: str | None = None, psm: int = 6, oem: int = 1):
        import pytesseract
        self.pytesseract = pytesseract

        tcfg = cfg["tesseract_ocr"]

        self.IMG_EXTS: Tuple[str, ...] = tuple(tcfg["image_extensions"])
        self.PDF_EXT: str = tcfg["pdf_extension"]
        self.PDF_DPI: int = int(tcfg["pdf_dpi"])
        self.LANG: str = tcfg["lang"]
        self.MAX_DIM: int = int(tcfg["max_dim"])

        tess_cmd = tcfg.get("tesseract_cmd")
        if isinstance(tess_cmd, str) and tess_cmd.strip():
            self.pytesseract.pytesseract.tesseract_cmd = tess_cmd

        self._tess_config = f"--psm {psm} --oem {oem}"

        self.poppler_path = poppler_path

    # ---------------- internal helpers ---------------- #

    def _resize_max_dim(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if max(h, w) <= self.MAX_DIM:
            return img_bgr
        scale = self.MAX_DIM / float(max(h, w))
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _binarize_otsu(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    def _ocr_image_array(self, arr_bgr: np.ndarray) -> Tuple[str, List[int]]:
        arr_bgr = self._resize_max_dim(arr_bgr)
        proc = self._binarize_otsu(arr_bgr)

        data = self.pytesseract.image_to_data(
            proc,
            lang=self.LANG,
            output_type=self.pytesseract.Output.DICT)

        texts = [t for t in data["text"] if str(t).strip()]
        confs = [int(c) for c in data["conf"] if str(c) != "-1"]
        return " ".join(texts), confs

    def ocr_path(self, path: Path) -> Tuple[str, List[int]]:
        """
        OCR a single path (image or PDF) with the exact legacy behavior:
          - images → space-joined tokens
          - PDFs → per-page text, joined with '\n\n'
        """
        ext = path.suffix.lower()

        if ext in self.IMG_EXTS:
            img = cv2.imread(str(path))
            if img is None:
                return "", []
            return self._ocr_image_array(img)

        if ext == self.PDF_EXT:
            pages = convert_from_path(
                str(path),
                dpi=self.PDF_DPI,
                poppler_path=self.poppler_path
            )
            chunks: List[str] = []
            confs_all: List[int] = []
            for pg in pages:
                arr = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
                text, confs = self._ocr_image_array(arr)
                chunks.append(text)
                confs_all.extend(confs)
            return "\n\n".join(chunks), confs_all

        return "", []
