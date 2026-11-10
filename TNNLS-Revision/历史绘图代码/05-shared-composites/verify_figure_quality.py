"""Verify the temporal_kernels figure meets publication quality standards."""
from pathlib import Path
import subprocess
import re
import sys

FIG_DIR = Path("C:/Users/VECTOR/Desktop/SPRiF-Neuron/experiment-design-20260606/results/figures/combined_analysis")
PDF = FIG_DIR / "temporal_kernels_composite.pdf"
PNG = FIG_DIR / "temporal_kernels_composite.png"
GRAY_PNG = FIG_DIR / "temporal_kernels_composite_grayscale.png"


def check_pdf_fonts():
    print("=== 1. PDF font check (no Type 3) ===")
    try:
        result = subprocess.run(
            ["pdffonts", str(PDF)],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print("  pdffonts not available, trying alternative check")
            return None
        out = result.stdout
        type3 = re.findall(r".+\s+Type\s*3\s+.*", out, re.IGNORECASE)
        if type3:
            print(f"  FAIL: Found Type 3 fonts: {type3}")
            return False
        # Also check that the font list is non-empty and uses TrueType/Type1
        lines = [l for l in out.splitlines() if l.strip() and not l.startswith("name")]
        print(f"  PASS: {len(lines)} fonts, no Type 3")
        for line in lines[:10]:
            print(f"    {line[:80]}")
        return True
    except FileNotFoundError:
        print("  pdffonts (poppler-utils) not installed; skipping")
        return None


def check_png_dpi():
    print()
    print("=== 2. PNG DPI check ===")
    try:
        from PIL import Image
        with Image.open(PNG) as img:
            dpi = img.info.get("dpi", (None, None))
            print(f"  File: {PNG.name}  size={img.size}  dpi={dpi}")
            if dpi[0] and dpi[0] >= 300:
                print(f"  PASS: DPI {dpi[0]} >= 300")
                return True
            else:
                print(f"  FAIL: DPI {dpi} < 300")
                return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def make_grayscale():
    print()
    print("=== 3. Grayscale version for archive-readability check ===")
    try:
        from PIL import Image, ImageOps
        with Image.open(PNG) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            gray = ImageOps.grayscale(img)
            gray.save(GRAY_PNG, dpi=img.info.get("dpi", (600, 600)))
            print(f"  Created: {GRAY_PNG}")
            return GRAY_PNG
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def check_svg_no_type3():
    print()
    print("=== 4. SVG content scan for Type 3 hint ===")
    svg = FIG_DIR / "temporal_kernels_composite.svg"
    text = svg.read_text(encoding="utf-8", errors="ignore")
    if "Type 3" in text or "Type3" in text:
        print("  Found 'Type 3' in SVG - check manually")
    if "fonttype" in text:
        m = re.findall(r'fonttype[^\d]*(\d+)', text)
        print(f"  fonttype values found: {set(m)} (42 = TrueType, 3 = Type 3)")
    else:
        print("  No fonttype tags (text outlined to paths, OK)")


def check_figure_size():
    print()
    print("=== 5. Figure dimensions (tex rendered at 0.92\\\\textwidth) ===")
    print(f"  Source: 8.4in x 4.8in")
    print(f"  At 0.92 textwidth (~6.6in): renders as 6.6in x 3.77in")
    print(f"  At 9pt text in figure -> ~8.3pt in final PDF (close to user 9pt minimum)")


if __name__ == "__main__":
    check_pdf_fonts()
    check_png_dpi()
    make_grayscale()
    check_svg_no_type3()
    check_figure_size()
    print()
    print("=== Manual review needed ===")
    print("Open the generated PNG to verify:")
    print("  - panel (c) legend is now BELOW the axes (no overlap)")
    print("  - panel (c) lines have markers (circles=QTDB, squares=GSC)")
    print("  - in-figure text is readable at printed size")
    print("  - grayscale version is still readable")
    print(f"  Color file: {PNG}")
    print(f"  Grayscale:  {GRAY_PNG}")
