
"""Experiment 4: analyze a Raman spectrum and save figures plus peak CSV."""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    detect_peaks,
    extract_peak_features,
)


def main():
    project_dir = Path(__file__).resolve().parent.parent
    file_path = project_dir / "data" / "raw" / "sample_raw.txt"
    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read, denoise, and baseline-correct the spectrum.
    spectrum = load_spectrum(file_path)
    denoised = denoise_spectrum(spectrum, window_length=9, polyorder=3)
    corrected = baseline_correct_spectrum(denoised, lam=1e6, p=0.01)

    x = corrected.spectral_axis
    y = corrected.spectral_data
    print(f"成功读取 {len(x)} 个光谱数据点")

    # 2. Compare detection thresholds (diagnostics only).
    print("\n===== 不同突出度阈值 =====")
    for threshold in (200, 500, 1000):
        test_peaks, _ = detect_peaks(
            corrected, prominence=threshold, distance=10
        )
        print(f"prominence={threshold}：{len(test_peaks)} 个候选峰")

    # 3. Final detection and feature extraction.
    peaks, properties = detect_peaks(
        corrected, prominence=1000, distance=10
    )
    features = extract_peak_features(corrected, peaks, properties)

    print("\n===== 正式峰特征 =====")
    for item in features:
        print(
            f"峰 {item['Peak_ID']}: "
            f"峰位={item['Position_cm-1']:.2f} cm^-1, "
            f"强度={item['Intensity']:.2f}, "
            f"突出度={item['Prominence']:.2f}, "
            f"半突出度宽度={item['Half_Prominence_Width_cm-1']:.2f} cm^-1"
        )

    # 4. Plot detected peaks.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y, color="blue", label="Corrected Raman Spectrum")
    ax.scatter(x[peaks], y[peaks], color="red", s=45,
               zorder=3, label="Detected Peaks")
    ax.set(title="Raman Peak Detection", xlabel="Raman Shift (cm$^{-1}$)",
           ylabel="Intensity (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_peaks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5. Plot half-prominence widths.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color="blue", label="Corrected Raman Spectrum")
    ax.scatter(x[peaks], y[peaks], color="red", s=45,
               zorder=3, label="Detected Peaks")
    for i, item in enumerate(features):
        ax.hlines(
            y=item["Width_Height"],
            xmin=item["Left_cm-1"],
            xmax=item["Right_cm-1"],
            color="orange", linewidth=3,
            label="Width at Half Prominence" if i == 0 else None,
        )
        ax.annotate(
            f"{item['Half_Prominence_Width_cm-1']:.1f}",
            xy=(item["Position_cm-1"], item["Intensity"]),
            xytext=(5, 12), textcoords="offset points",
            fontsize=9, color="darkred",
        )
    ax.set(title="Raman Peak Width Analysis", xlabel="Raman Shift (cm$^{-1}$)",
           ylabel="Intensity (a.u.)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_peak_widths.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # 6. Export five core feature columns.
    csv_path = output_dir / "04_peak_features.csv"
    fieldnames = [
        "Peak_ID",
        "Position_cm-1",
        "Intensity",
        "Prominence",
        "Half_Prominence_Width_cm-1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for item in features:
            writer.writerow(item)

    print(f"\n正式检测到 {len(features)} 个候选峰")
    print("输出目录：", output_dir)
    print("文件：04_peaks.png、04_peak_widths.png、04_peak_features.csv")


if __name__ == "__main__":
    main()
