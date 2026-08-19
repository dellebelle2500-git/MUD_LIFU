#!/usr/bin/env python3
"""Build all Nature-ready main and Extended Data figures.

Run from any directory after installing requirements.txt. Figure 4 imports the
official PyPI ``phate`` package; there is intentionally no local phate.py.
"""
from pathlib import Path
import subprocess
import sys
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

MAIN = [f"Figure{i}_final.py" for i in range(1, 7)]
EXT = [f"ExtDataFig{i}_final.py" for i in range(1, 5)]


def run(script_name: str) -> None:
    print(f"[build] {script_name}")
    subprocess.run([sys.executable, str(SCRIPTS / script_name)], cwd=ROOT, check=True)


def merge_pdf(paths, output):
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(str(path))
        for page in reader.pages:
            writer.add_page(page)
    with open(output, "wb") as fh:
        writer.write(fh)


def main():
    for script in MAIN + EXT:
        run(script)

    # Convenience review composites. Individual files are the submission units.
    merge_pdf([OUT / f"Figure{i}.pdf" for i in range(1, 7)],
              OUT / "MUD_main_Figs_NatureReady.pdf")
    merge_pdf([OUT / f"ExtDataFig{i}.pdf" for i in range(1, 5)],
              OUT / "MUD_ExtData_Figs_NatureReady.pdf")

    print("\nBuilt submission artwork:")
    for i in range(1, 7):
        print(f"  {OUT / f'Figure{i}.pdf'}")
    for i in range(1, 5):
        print(f"  {OUT / f'ExtDataFig{i}.jpg'}")
    print("\nReview composites:")
    print(f"  {OUT / 'MUD_main_Figs_NatureReady.pdf'}")
    print(f"  {OUT / 'MUD_ExtData_Figs_NatureReady.pdf'}")


if __name__ == "__main__":
    main()
