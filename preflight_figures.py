#!/usr/bin/env python3
"""Preflight generated PDFs against the typography/size rules used here."""
from pathlib import Path
import re
import sys
import fitz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MM_PER_PT = 25.4 / 72.0
EXPECTED_WIDTH_MM = 180.0
MAX_HEIGHT_MM = 170.0
SANS_TOKENS = ("Arial", "Helvetica", "NimbusSans", "Arimo", "LiberationSans", "DejaVuSans")
PANEL_RE = re.compile(r"^[a-l]$")
MATH_SCRIPT_RE = re.compile(r"^(?:10|[+\-−]?\d{1,3})$")


def preflight(path: Path):
    errors, warnings = [], []
    doc = fitz.open(path)
    if len(doc) != 1:
        errors.append(f"expected one page, found {len(doc)}")
        return errors, warnings
    page = doc[0]
    wmm, hmm = page.rect.width * MM_PER_PT, page.rect.height * MM_PER_PT
    if abs(wmm - EXPECTED_WIDTH_MM) > 0.15:
        errors.append(f"width {wmm:.2f} mm (expected {EXPECTED_WIDTH_MM:.1f})")
    if hmm > MAX_HEIGHT_MM + 0.15:
        errors.append(f"height {hmm:.2f} mm exceeds {MAX_HEIGHT_MM:.1f}")

    spans = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    spans.append((text, float(span["size"]), span.get("font", "")))

    for text, size, font in spans:
        is_panel = PANEL_RE.fullmatch(text) and 7.7 <= size <= 8.3
        if size > 7.05 and not is_panel:
            errors.append(f"text >7 pt: {text!r} ({size:.2f} pt)")
        if size < 4.95:
            # Matplotlib mathtext renders superscript exponents as smaller glyphs.
            if MATH_SCRIPT_RE.fullmatch(text):
                warnings.append(f"math superscript below 5 pt: {text!r} ({size:.2f} pt)")
            else:
                errors.append(f"text <5 pt: {text!r} ({size:.2f} pt)")
        if font and not any(tok.lower() in font.lower() for tok in SANS_TOKENS):
            errors.append(f"non-standard/suspect font {font!r} for {text!r}")

    return errors, warnings


def main():
    files = [OUT / f"Figure{i}.pdf" for i in range(1, 7)] + \
            [OUT / f"ExtDataFig{i}.pdf" for i in range(1, 5)]
    missing = [p for p in files if not p.exists()]
    if missing:
        print("Missing outputs; run python build_all.py first:")
        for p in missing:
            print(" ", p)
        return 2

    any_error = False
    for p in files:
        errors, warnings = preflight(p)
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {p.name}")
        for w in warnings:
            print("  warning:", w)
        for e in errors:
            print("  error:", e)
        any_error |= bool(errors)
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
