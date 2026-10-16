
from pathlib import Path
import numpy as np

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    detect_peaks,
    extract_peak_features
)


# 1. 读取同一条原始光谱
project_dir = Path(__file__).resolve().parent.parent

file_path = project_dir / "data/raw/sample_raw.txt"

spectrum = load_spectrum(file_path)


# 2. 历史实验记录
historical_positions = np.array([
    267.41, 388.15, 534.34, 667.59,
    714.98, 765.54, 934.41, 1018.06
])

historical_widths = np.array([
    18.21, 81.22, 69.20, 133.09,
    18.04, 30.38, 74.33, 94.10
])


# 3. 比较 SG 窗口 9 和 21
for window in [9, 21]:

    print("\n" + "=" * 45)
    print(f"SG 窗口长度：{window}")
    print("=" * 45)

    # SG 去噪
    denoised = denoise_spectrum(
        spectrum,
        window_length=window,
        polyorder=3
    )

    # AsLS 基线校正
    corrected = baseline_correct_spectrum(
        denoised,
        lam=1e6,
        p=0.01
    )

    # 自动寻峰
    peaks, properties = detect_peaks(
        corrected,
        prominence=1000,
        distance=10
    )

    # 提取峰特征
    features = extract_peak_features(
        corrected,
        peaks,
        properties
    )

    positions = np.array([
        f["Position_cm-1"]
        for f in features
    ])

    widths = np.array([
        f["Half_Prominence_Width_cm-1"]
        for f in features
    ])

    print("候选峰数量：", len(features))

    print("峰位：")
    print(np.round(positions, 2))

    print("半突出度宽度：")
    print(np.round(widths, 2))

    # 检查是否复现历史记录
    if len(features) == len(historical_positions):

        positions_match = np.allclose(
            positions,
            historical_positions,
            atol=0.02,
            rtol=0
        )

        widths_match = np.allclose(
            widths,
            historical_widths,
            atol=0.02,
            rtol=0
        )

        print("历史峰位一致：", positions_match)
        print("历史峰宽一致：", widths_match)

    else:
        print("峰数量与历史记录不同")
