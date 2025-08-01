import os
import random
import yaml
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter

#--- main function ---#
def imitate_scans(
    mode: str = "default",
    level: str = "level2",
    input_folder: str = None,
    output_folder: str = None):

    """
    Convert PDFs to 'scans' (JPEGs) with augmentations.

    mode:
      - "default": read from data/raw/<subfolders>, write to
                   data/scans/docs & technical_drawings
      - "test":    flat: read all PDFs from one folder and overwrite
                   them in-place with JPEGs

    level:
      - "level1" / "level2" / "level3"  -> selects augmentation strength block from YAML

    input_folder/output_folder override config paths if provided.
    """

    # --- load config.yaml --- #
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    ROOT = os.path.abspath(cfg["paths"]["project_root"])
    POPPLER = cfg["paths"]["poppler"]["bin_path"]

    try:
        pdf_cfg  = cfg["augmentation"]["levels"][level]["pdf"]
        proc_cfg = cfg["augmentation"]["levels"][level]["process_image"]
    except KeyError:
        raise KeyError(f"Level '{level}' not found under augmentation.levels in config.yaml")

    if mode == "default":
        base_in  = input_folder or os.path.join(ROOT, cfg["paths"]["data"]["raw"])
        out_docs = output_folder or os.path.join(
            ROOT, cfg["paths"]["data"]["scans"]["docs"][level])
        
    elif mode == "test":
        base_in  = input_folder or os.path.join(ROOT, cfg["generator_settings"]["test"]["output_folder"])
        out_docs = output_folder or base_in
    else:
        raise ValueError("mode must be 'default' or 'test'")

    os.makedirs(out_docs, exist_ok=True)

    # ---------------------- Helper functions ---------------------- #
    def add_noise(img, intensity):
        noise = np.random.normal(0, intensity, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_vertical_streak(pil_img):
        img_np = np.array(pil_img).astype(np.float32)
        h, w, _ = img_np.shape
        x0 = random.randint(int(w * 0.2), int(w * 0.8))
        width = random.randint(proc_cfg["vertical_streak_width_min"],
                               proc_cfg["vertical_streak_width_max"])
        intensity = random.uniform(proc_cfg["vertical_streak_intensity_min"],
                                   proc_cfg["vertical_streak_intensity_max"])
        for x in range(x0, min(w, x0 + width)):
            alpha = 1 - abs((x - x0) / width)
            img_np[:, x, :] *= (1 - alpha * (1 - intensity))
        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def add_dust(pil_img, count, max_radius):
        arr = np.array(pil_img)
        h, w = arr.shape[:2]
        for _ in range(count):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            r = random.randint(1, max_radius)
            color = random.choice([0, 255])
            cv2.circle(arr, (x, y), r, (color, ) * 3, -1)
        return Image.fromarray(arr)

    def apply_motion_blur(img, size, direction):
        if direction == "horizontal":
            kernel = np.ones((1, size), np.float32) / size
        else:
            kernel = np.ones((size, 1), np.float32) / size
        return cv2.filter2D(img, -1, kernel)

    def apply_page_warp(pil_img):
        arr = np.array(pil_img).astype(np.float32)
        h, w = arr.shape[:2]
        amp = random.uniform(*proc_cfg["page_warp_amplitude"])
        wave = np.sin(np.linspace(0, 2 * np.pi, h)) * amp
        ys = np.arange(h)
        xs = np.arange(w)
        map_x = (xs + wave[:, None]).astype(np.float32)
        map_y = (ys[:, None] @ np.ones((1, w))).astype(np.float32)
        warped = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
        return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))

    def apply_crumple_map(pil_img):
        arr = np.array(pil_img).astype(np.float32)
        h, w = arr.shape[:2]
        bumps = np.random.normal(0, proc_cfg["crumple_map_std"], (h, w))
        bumps = gaussian_filter(bumps, sigma=proc_cfg["crumple_map_blur_kernel"][0])
        map_x = (np.tile(np.arange(w), (h, 1)) + bumps).astype(np.float32)
        map_y = (np.tile(np.arange(h), (w, 1)).T + bumps).astype(np.float32)
        warped = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))

    def add_vignette(pil_img, strength):
        arr = np.array(pil_img).astype(np.float32)
        h, w = arr.shape[:2]
        ys = (np.arange(h) - h / 2) / (h / 2)
        xs = (np.arange(w) - w / 2) / (w / 2)
        mask = 1 - strength * np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        mask = np.clip(mask, 1 - strength, 1)[:, :, None]
        return Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))

    def add_inverse_vignette(pil_img, strength):
        arr = np.array(pil_img).astype(np.float32)
        h, w = arr.shape[:2]
        ys = (np.arange(h) - h / 2) / (h / 2)
        xs = (np.arange(w) - w / 2) / (w / 2)
        mask = 1 + strength * (1 - np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2))
        mask = np.clip(mask, 1, 1 + strength)[:, :, None]
        return Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))

    def add_paper_texture(pil_img):
        paths    = proc_cfg["paper_texture_paths"]
        opacity  = proc_cfg["paper_texture_opacity"]
        scale    = proc_cfg["paper_texture_scale"]

        tex = Image.open(random.choice(paths)).convert("L")
        tex = tex.resize((int(pil_img.width * scale), int(pil_img.height * scale)),
                         resample=Image.LANCZOS)
        tex = ImageOps.fit(tex, (pil_img.width, pil_img.height), method=Image.LANCZOS)

        base    = pil_img.convert("RGBA")
        alpha   = tex.point(lambda p: int(p * opacity))
        overlay = Image.merge("RGBA", (tex, tex, tex, alpha))
        return Image.alpha_composite(base, overlay).convert("RGB")

    def process_image(pil_img):
        pil = pil_img

        # resize
        pil = pil.resize((int(pil.width * proc_cfg["resize_scale"]),
                          int(pil.height * proc_cfg["resize_scale"])),
                         Image.BICUBIC)

        # contrast
        if random.random() < proc_cfg["contrast_probability"]:
            lo, hi = proc_cfg["contrast_range"]
            pil = ImageEnhance.Contrast(pil).enhance(random.uniform(lo, hi))

        # brightness
        if random.random() < proc_cfg["brightness_probability"]:
            lo, hi = proc_cfg["brightness_range"]
            pil = ImageEnhance.Brightness(pil).enhance(random.uniform(lo, hi))

        # crop
        if random.random() < proc_cfg["crop_probability"]:
            dx, dy = proc_cfg["crop_max_offset"]
            pil = pil.crop((random.randint(0, dx),
                            random.randint(0, dy),
                            pil.width,
                            pil.height))

        # rotate small angle
        if random.random() < proc_cfg["rotate_probability"]:
            lo, hi = proc_cfg["rotate_angle_range"]
            pil = pil.rotate(random.uniform(lo, hi), expand=True, fillcolor="white")

        # rotate 180
        if random.random() < proc_cfg["rotate_180_probability"]:
            pil = pil.rotate(180)

        # noise
        if random.random() < proc_cfg["noise_probability"]:
            lo, hi = proc_cfg["noise_std_range"]
            std = random.uniform(lo, hi)
            pil = Image.fromarray(add_noise(np.array(pil), std))

        # vertical streak
        if random.random() < proc_cfg["vertical_streak_probability"]:
            pil = add_vertical_streak(pil)

        # dust
        if random.random() < proc_cfg["dust_probability"]:
            pil = add_dust(pil, proc_cfg["dust_count"], proc_cfg["dust_max_radius"])

        # motion blur
        if random.random() < proc_cfg["motion_blur_probability"]:
            size = random.choice(proc_cfg["motion_blur_size_options"])
            direction = random.choice(proc_cfg["motion_blur_directions"])
            pil = Image.fromarray(apply_motion_blur(np.array(pil), size, direction))

        # page warp
        if random.random() < proc_cfg["page_warp_probability"]:
            pil = apply_page_warp(pil)

        # crumple map
        if random.random() < proc_cfg["crumple_map_probability"]:
            pil = apply_crumple_map(pil)

        # vignette
        if random.random() < proc_cfg["vignette_probability"]:
            pil = add_vignette(pil, proc_cfg["vignette_strength"])

        # inverse vignette
        if random.random() < proc_cfg.get("inverse_vignette_probability", 0):
            pil = add_inverse_vignette(pil, proc_cfg["vignette_strength"])

        # paper texture
        if random.random() < proc_cfg["paper_texture_probability"]:
            pil = add_paper_texture(pil)

        return pil

    # ---------------------- Processing ---------------------- #
    print(f"=== Start scan imitation ({mode}, {level}) ===")

    # test mode
    if mode == "test":
        pdfs = [f for f in os.listdir(base_in) if f.lower().endswith(".pdf")]
        for pdf in pdfs:
            src = os.path.join(base_in, pdf)
            try:
                pages = convert_from_path(src, dpi=pdf_cfg["dpi"], poppler_path=POPPLER)
            except Exception as e:
                print(f"   conversion error {pdf}: {e}")
                continue

            for idx, page in enumerate(pages):
                img = process_image(page)
                name = f"{os.path.splitext(pdf)[0]}_p{idx}.jpg"
                img.save(
                    os.path.join(base_in, name),
                    "JPEG",
                    quality=random.randint(*proc_cfg["jpeg_quality_range"]))
                
        print(f"=== Scan imitation ({mode}) complete ===")
        return

    # default mode (walk each subfolder in raw/)
    for sub in os.listdir(base_in):
        in_f = os.path.join(base_in, sub)
        if not os.path.isdir(in_f):
            continue

        out_sub = os.path.join(out_docs, sub)
        os.makedirs(out_sub, exist_ok=True)

        pdfs = [f for f in os.listdir(in_f) if f.lower().endswith(".pdf")]
        for pdf in pdfs:
            src = os.path.join(in_f, pdf)
            try:
                pages = convert_from_path(src, dpi=pdf_cfg["dpi"], poppler_path=POPPLER)
                print(f"  → converted {pdf}, pages: {len(pages)}")
            except Exception as e:
                print(f"   conversion error {pdf}: {e}")
                continue

            for idx, page in enumerate(pages):
                img = process_image(page)
                name = f"{os.path.splitext(pdf)[0]}_p{idx}.jpg"
                img.save(
                    os.path.join(out_sub, name),
                    "JPEG",
                    quality=random.randint(*proc_cfg["jpeg_quality_range"])
                )
        print(f"  >>>> saved {len(pdfs)} page(s) for '{sub}'")

    print(f"=== Scan imitation ({mode}, {level}) complete ===")

if __name__ == "__main__":
    imitate_scans(mode="default", level="level3")