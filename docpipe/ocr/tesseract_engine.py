import os
import cv2
import numpy as np
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

from typing import Iterable, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TesseractConfig:
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    pdf_extension: str = ".pdf"
    pdf_dpi: int = 300
    lang: str = "eng"
    max_dim: int = 3000
    tesseract_cmd: Optional[str] = None

class TesseractOCREngine:
    def __init__(
        self,
        cfg: dict,
        poppler_path: Optional[str] = None,
        psm: int = 6,
        oem: int = 1,
        dpi: int = 300,
        fallback_psms: Iterable[int] = (4, 3, 11, 12, 6),
        whitelist: Optional[str] = None,
    ) -> None:
        import pytesseract as _pt
        tcfg_raw = cfg.get("tesseract_ocr", {}) if isinstance(cfg, dict) else {}
        tcfg = TesseractConfig(
            image_extensions=tuple(tcfg_raw.get("image_extensions", TesseractConfig.image_extensions)),
            pdf_extension=str(tcfg_raw.get("pdf_extension", TesseractConfig.pdf_extension)),
            pdf_dpi=int(tcfg_raw.get("pdf_dpi", TesseractConfig.pdf_dpi)),
            lang=str(tcfg_raw.get("lang", TesseractConfig.lang)),
            max_dim=int(tcfg_raw.get("max_dim", TesseractConfig.max_dim)),
            tesseract_cmd=tcfg_raw.get("tesseract_cmd"),
        )

        self.pytesseract = _pt
        if isinstance(tcfg.tesseract_cmd, str) and tcfg.tesseract_cmd.strip():
            self.pytesseract.pytesseract.tesseract_cmd = tcfg.tesseract_cmd

        self.IMG_EXTS: Tuple[str, ...] = tcfg.image_extensions
        self.PDF_EXT: str = tcfg.pdf_extension.lower()
        self.PDF_DPI: int = int(dpi if dpi is not None else tcfg.pdf_dpi)
        self.LANG: str = tcfg.lang
        self.MAX_DIM: int = tcfg.max_dim
        self.poppler_path = poppler_path

        self._base_tess_config = f"--oem {int(oem)} --dpi {int(self.PDF_DPI)} -c preserve_interword_spaces=1"
        if whitelist:
            self._base_tess_config += f" -c tessedit_char_whitelist={whitelist}"
        self._psm_candidates: List[int] = self._unique_ordered([psm, *fallback_psms])

    @staticmethod
    def _unique_ordered(values: Iterable[int]) -> List[int]:
        seen, out = set(), []
        for v in values:
            if v not in seen:
                out.append(v); seen.add(v)
        return out

    def _resize_max_dim(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if max(h, w) <= self.MAX_DIM:
            return img_bgr
        s = self.MAX_DIM / float(max(h, w))
        return cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    def _ensure_min_side(self, gray: np.ndarray, target_max: int = 1400) -> np.ndarray:
        H, W = gray.shape[:2]
        if max(H, W) >= target_max:
            return gray
        s = target_max / float(max(H, W))
        return cv2.resize(gray, (int(W*s), int(H*s)), interpolation=cv2.INTER_CUBIC)

    def _deskew_osd(self, gray: np.ndarray) -> np.ndarray:
        try:
            osd = self.pytesseract.image_to_osd(gray, output_type=self.pytesseract.Output.DICT)
            angle = float(osd.get("rotate", 0) or 0)
        except Exception:
            angle = 0.0
        if abs(angle) > 0.1:
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return gray

    def _prep(self, img_bgr: np.ndarray) -> np.ndarray:
        img_bgr = self._resize_max_dim(img_bgr)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        gray = self._ensure_min_side(gray, target_max=1400)
        gray = self._deskew_osd(gray)

        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            31, 11,
        )
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((1,1), np.uint8), iterations=1)
        return thr

    def _image_to_data_with_psm(self, img_bin: np.ndarray, psm: int):
        cfg = f"{self._base_tess_config} --psm {int(psm)}"
        return self.pytesseract.image_to_data(
            img_bin, lang=self.LANG, config=cfg,
            output_type=self.pytesseract.Output.DICT,
        )

    @staticmethod
    def _extract_texts_and_confs(data_dict):
        texts, confs = [], []
        for t, c in zip(data_dict.get("text", []), data_dict.get("conf", [])):
            t = str(t) if t is not None else ""
            if t.strip():
                texts.append(t.strip())
            if str(c) != "-1":
                try: confs.append(int(float(c)))
                except Exception: pass
        return " ".join(texts), confs

    @staticmethod
    def _avg_conf(confs):
        return float(np.mean(confs)) if confs else 0.0

    def _ocr_image_array(self, arr_bgr: np.ndarray):
        proc = self._prep(arr_bgr)

        best_text, best_confs, best_score = "", [], -1.0
        for psm in self._psm_candidates:
            try:
                data = self._image_to_data_with_psm(proc, psm)
                text, confs = self._extract_texts_and_confs(data)
                score = self._avg_conf(confs)
                if score > best_score:
                    best_text, best_confs, best_score = text, confs, score
            except Exception:
                continue
        return best_text, best_confs

    def ocr_path(self, path: Path):
        ext = path.suffix.lower()
        if ext in self.IMG_EXTS:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return "", []
            return self._ocr_image_array(img)
        if ext == self.PDF_EXT:
            pages = convert_from_path(str(path), dpi=self.PDF_DPI, poppler_path=self.poppler_path)
            chunks, confs_all = [], []
            for pg in pages:
                arr_bgr = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
                text, confs = self._ocr_image_array(arr_bgr)
                chunks.append(text); confs_all.extend(confs)
            return "\n\n".join(chunks), confs_all
        return "", []


def get_ocr_function(engine, cfg):
    img_exts = tuple(cfg[engine]["image_extensions"])
    pdf_ext = cfg[engine]["pdf_extension"]
    poppler = cfg["paths"]["poppler"]["bin_path"]

    if engine == "easy_ocr":
        import easyocr
        lang = cfg[engine]["lang"]
        max_dim = cfg[engine]["max_dim"]
        reader = easyocr.Reader([lang] if isinstance(lang, str) else lang, gpu=False)
        def ocr_easyocr(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in img_exts:
                import cv2
                img = cv2.imread(path)
                if img is None: return ""
                h, w = img.shape[:2]
                if max(h, w) > max_dim:
                    s = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w*s), int(h*s)))
                return "\n".join(reader.readtext(img, detail=0))
            elif ext == pdf_ext:
                from pdf2image import convert_from_path
                pages = convert_from_path(path, dpi=cfg[engine]["pdf_dpi"], poppler_path=poppler)
                out = ["\n".join(reader.readtext(np.array(pg)[:, :, ::-1], detail=0)) for pg in pages]
                return "\n\n".join(out)
            return ""
        return ocr_easyocr

    elif engine == "tesseract_ocr":
        if "tesseract_cmd" in cfg["tesseract_ocr"]:
            pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_ocr"]["tesseract_cmd"]

        tess = TesseractOCREngine(
            cfg=cfg,
            poppler_path=poppler,
            psm=6, oem=1,
            dpi=cfg["tesseract_ocr"]["pdf_dpi"],
            fallback_psms=(4, 3, 11, 12, 6),
            whitelist=None,
        )

        def ocr_tesseract(path):
            text, _confs = tess.ocr_path(Path(path))
            return text.strip()

        return ocr_tesseract

    else:
        raise ValueError(f"Unknown OCR engine: {engine}")
