from pathlib import Path
import shutil
import sys

# === ŚCIEŻKI ===
SOURCE_DIR = Path(r"C:\Users\janek\Documents\GitHub\document_classificator\tests\test_data")
DEST_DIR   = Path(r"C:\Users\janek\Documents\GitHub\document_classificator\tests\input")

def copy_x_to_y(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Źródło nie istnieje lub nie jest katalogiem: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)

def main():
    try:
        copy_x_to_y(SOURCE_DIR, DEST_DIR)
        print(f"[Kopiowanie] Zakończono kopiowanie z {SOURCE_DIR} do {DEST_DIR}")
    except Exception as e:
        print(f"[Kopiowanie] Błąd: {e}", file=sys.stderr)
        sys.exit(1)  

    from docpipe.cli.pipeline_main import run_main_loop
    run_main_loop()

if __name__ == "__main__":
    main()
