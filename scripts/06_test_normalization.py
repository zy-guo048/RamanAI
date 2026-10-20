
"""
RamanAI 实验 6.10

验证自定义最大值归一化函数。
"""

from pathlib import Path

import numpy as np

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    normalize_spectrum
)


def main():

    # ======================================
    # 1. 定位原始 Spinel 光谱
    # ======================================

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "fair_oriented"

    matches = sorted(
        file
        for file in data_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() == ".txt"
        and file.name.startswith("Spinel__R050259-3__")
        and "raman_data_raw" in file.name.lower()
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"预期找到 1 条光谱，实际找到 {len(matches)} 条"
        )

    file_path = matches[0]
    print("测试光谱：", file_path.name)

    # ======================================
    # 2. 读取并预处理光谱
    # ======================================

    spectrum = load_spectrum(file_path)

    denoised = denoise_spectrum(
        spectrum,
        window_length=21,
        polyorder=3
    )

    corrected = baseline_correct_spectrum(
        denoised,
        lam=1e6,
        p=0.01
    )

    # ======================================
    # 3. 旧方法：直接使用 NumPy
    # ======================================

    y = corrected.spectral_data

    old_normalized_y = y / np.max(y)

    # ======================================
    # 4. 新方法：调用自定义函数
    # ======================================

    normalized = normalize_spectrum(
        corrected,
        method="max"
    )

    new_normalized_y = normalized.spectral_data

    # ======================================
    # 5. 检查新旧方法是否一致
    # ======================================

    np.testing.assert_allclose(
        old_normalized_y,
        new_normalized_y,
        rtol=0,
        atol=1e-12
    )

    print("新旧归一化结果一致：True")

    # ======================================
    # 6. 检查最大强度
    # ======================================

    maximum = np.max(new_normalized_y)

    print("归一化后最大强度：", maximum)

    assert np.isclose(maximum, 1.0)

    # ======================================
    # 7. 检查波数轴是否改变
    # ======================================

    np.testing.assert_array_equal(
        corrected.spectral_axis,
        normalized.spectral_axis
    )

    print("拉曼位移轴保持不变：True")

    # ======================================
    # 8. 检查最强峰位置是否改变
    # ======================================

    old_index = np.argmax(y)
    new_index = np.argmax(new_normalized_y)

    assert old_index == new_index

    print(
        "最强峰位置保持不变：True"
    )

    print(
        "最强峰位置：",
        normalized.spectral_axis[new_index],
        "cm^-1"
    )

    print("\n全部归一化测试通过！")


if __name__ == "__main__":
    main()
