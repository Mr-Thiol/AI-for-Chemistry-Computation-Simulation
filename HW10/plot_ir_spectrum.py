from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


def read_ir_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frequencies = []
    intensities = []

    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            frequencies.append(float(row[0]))
            intensities.append(float(row[1]))

    return np.array(frequencies), np.array(intensities)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "IR_Spectrum_2026-06-08T02-02-06.txt"

    freq, intensity = read_ir_data(data_file)

    # Avoid log(0) for absorbance conversion.
    safe_intensity = np.clip(intensity, 1e-12, None)
    absorbance = -np.log10(safe_intensity)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    ax1.plot(freq, intensity, color="#1f77b4", linewidth=1.4, label="Intensity")
    ax1.set_xlabel("Frequency (cm$^{-1}$)")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Infrared Spectrum (Intensity)")
    ax1.legend(loc="upper right")
    fig1.text(
        0.5,
        0.01,
        "Figure 1. IR spectrum plotted as intensity versus frequency.",
        ha="center",
        fontsize=9,
    )
    fig1.tight_layout(rect=[0, 0.03, 1, 1])
    fig1.savefig(base_dir / "ir_spectrum_intensity.png")

    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    ax2.plot(freq, absorbance, color="#d62728", linewidth=1.4, label="Absorbance")
    ax2.set_xlabel("Frequency (cm$^{-1}$)")
    ax2.set_ylabel("Absorbance (A = -log10(I))")
    ax2.set_title("Infrared Spectrum (Absorbance)")
    ax2.legend(loc="upper right")
    fig2.text(
        0.5,
        0.01,
        "Figure 2. IR spectrum converted to absorbance from intensity.",
        ha="center",
        fontsize=9,
    )
    fig2.tight_layout(rect=[0, 0.03, 1, 1])
    fig2.savefig(base_dir / "ir_spectrum_absorbance.png")

    plt.close("all")


if __name__ == "__main__":
    main()
