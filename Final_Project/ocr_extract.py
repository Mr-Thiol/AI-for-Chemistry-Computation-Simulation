"""
OCR extraction script for Final_Project images.
Extracts Energy and Adsorption Energy fields from images using pytesseract.
"""

import os
import re
import glob
from PIL import Image
import pytesseract


def extract_fields_from_text(text):
    """
    Extract Energy and Adsorption Energy from OCR text.

    Expected patterns (as seen in the web app screenshots):
      Energy: -2493.3750 eV          <- adsorption structure energy
      Adsorption Energy: -3.357 eV (w.D3: -3.991 eV)

    The "Energy" that matters is the one immediately before the
    "Adsorption Energy" line in the image (center-panel energy),
    not the molecule energy that appears elsewhere in the image.
    We find the Adsorption Energy position first, then pick the
    last Energy match that occurs before it in the OCR text.
    """
    energy = None
    adsorption_energy = None

    # --- Adsorption Energy with D3 correction ---
    ads_pattern = re.compile(
        r'Adsorption\s+Energy\s*[:\s]\s*([-+]?\d+[.,]\d+)\s*eV\s*'
        r'\(?\s*w\.?\s*D3\s*[:\s]\s*([-+]?\d+[.,]\d+)\s*eV\s*\)?',
        re.IGNORECASE,
    )
    ads_match = ads_pattern.search(text)
    if ads_match:
        val1 = ads_match.group(1).replace(',', '.')
        val2 = ads_match.group(2).replace(',', '.')
        adsorption_energy = f"{val1} eV (w.D3: {val2} eV)"
        ads_pos = ads_match.start()
    else:
        # Fallback: Adsorption Energy without D3 part
        ads_simple = re.search(
            r'Adsorption\s+Energy\s*[:\s]\s*([-+]?\d+[.,]\d+)\s*eV',
            text, re.IGNORECASE,
        )
        if ads_simple:
            val = ads_simple.group(1).replace(',', '.')
            adsorption_energy = f"{val} eV"
            ads_pos = ads_simple.start()
        else:
            ads_pos = len(text)  # no limit

    # --- Stand-alone Energy: pick the LAST match before ads_pos ---
    # This avoids picking up the molecule energy in the top-left panel.
    energy_pattern = re.compile(
        r'(?<!Adsorption\s)(?<!Adsorption)Energy\s*[:\s]\s*([-+]?\d+[.,]\d+)\s*eV',
        re.IGNORECASE,
    )
    best_energy = None
    for m in energy_pattern.finditer(text):
        if m.start() < ads_pos:
            # Verify "Adsorption" does not appear within 20 chars before the match
            prefix = text[max(0, m.start() - 20): m.start()]
            if not re.search(r'Adsorption', prefix, re.IGNORECASE):
                best_energy = m.group(1).replace(',', '.')
        # Stop scanning once we pass the Adsorption Energy position
        if m.start() > ads_pos:
            break
    energy = best_energy

    return energy, adsorption_energy


def ocr_image(image_path):
    """
    Run pytesseract on a single image and return raw text.
    Also re-runs OCR on the bottom-center crop to improve capture of the
    'Energy / Adsorption Energy' lines that appear there.
    """
    img = Image.open(image_path)
    w, h = img.size

    # --- Full image (PSM 6: single uniform block of text) ---
    cfg6 = r'--oem 3 --psm 6'
    text_full = pytesseract.image_to_string(img, lang='eng+chi_sim', config=cfg6)

    # --- Bottom-centre crop: roughly where the summary text lives ---
    # x: 33% – 70% of width, y: 82% – 100% of height
    left   = int(w * 0.33)
    top    = int(h * 0.80)
    right  = int(w * 0.72)
    bottom = h
    crop = img.crop((left, top, right, bottom))

    # Scale up 2× to help OCR on small text
    crop_big = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    cfg11 = r'--oem 3 --psm 11'   # sparse text — good for mixed regions
    text_crop = pytesseract.image_to_string(crop_big, lang='eng+chi_sim', config=cfg11)

    return text_full + "\n" + text_crop


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(script_dir, ext)))
    image_files.sort()

    if not image_files:
        print("No image files found in", script_dir)
        return

    print(f"Found {len(image_files)} image(s). Running OCR with pytesseract...")

    output_path = os.path.join(script_dir, "ocr_results.txt")
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("Image Name, Energy (eV), Adsorption Energy\n")
        out_f.write("-" * 80 + "\n")

        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"Processing: {img_name}")
            try:
                raw_text = ocr_image(img_path)
                print(f"  OCR snippet: {raw_text[:300].strip()!r}")
                energy, adsorption_energy = extract_fields_from_text(raw_text)
                energy_str = energy if energy is not None else "N/A"
                ads_str = adsorption_energy if adsorption_energy is not None else "N/A"
                line = f"{img_name}, {energy_str}, {ads_str}\n"
                out_f.write(line)
                print(f"  -> Energy: {energy_str} | Adsorption Energy: {ads_str}")
            except Exception as e:
                print(f"  ERROR processing {img_name}: {e}")
                out_f.write(f"{img_name}, ERROR, ERROR\n")

    print(f"\nDone! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
