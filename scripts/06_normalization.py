
"""
RamanAI 实验 6.8

最大值归一化：
1. 读取真实 Spinel RAW 光谱
2. SG 去噪
3. AsLS 基线校正
4. 最大峰强归一化
5. 可视化并验证结果
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum
)


def main():

    # ======================================
    # 1. 设置路径
    # ======================================

    project_dir = Path(__file__).resolve().parent.parent

    data_dir = project_dir / "fair_oriented"

    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================
    # 2. 查找指定的 Spinel RAW 光谱
    # ======================================

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
            f"预期找到1条 Spinel 光谱，实际找到 {len(matches)} 条"
        )

    file_path = matches[0]

    print("正在分析：", file_path.name)

    # ======================================
    # 3. 读取和预处理光谱
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

    x = corrected.spectral_axis
    y = corrected.spectral_data

    # ======================================
    # 4. 最大值归一化
    # ======================================

    max_intensity = np.max(y)

    if not np.isfinite(max_intensity) or max_intensity <= 0:
        raise ValueError(
            "光谱最大强度必须是有限正数，无法归一化"
        )

    normalized_y = y / max_intensity

    # ======================================
    # 5. 检查计算结果
    # ======================================

    print("\n===== 归一化前 =====")
    print("最大强度：", max_intensity)
    print("最小强度：", np.min(y))

    print("\n===== 归一化后 =====")
    print("最大强度：", np.max(normalized_y))
    print("最小强度：", np.min(normalized_y))

    max_index = np.argmax(y)

    print(
        "最强峰所在波数：",
        x[max_index],
        "cm^-1"
    )

    print(
        "归一化是否成功：",
        np.isclose(np.max(normalized_y), 1.0)
    )

    # ======================================
    # 6. 绘制归一化前后对比图
    # ======================================

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True
    )

    # 上图：归一化前
    axes[0].plot(
        x,
        y,
        color="blue"
    )

    axes[0].set_title(
        "Before Normalization"
    )

    axes[0].set_ylabel(
        "Corrected Intensity (a.u.)"
    )

    axes[0].grid(alpha=0.25)

    # 下图：归一化后
    axes[1].plot(
        x,
        normalized_y,
        color="darkorange"
    )

    axes[1].set_title(
        "After Max Normalization"
    )

    axes[1].set_xlabel(
        "Raman Shift (cm^-1)"
    )

    axes[1].set_ylabel(
        "Normalized Intensity"
    )

    axes[1].grid(alpha=0.25)

    # 标出归一化后的最大值
    axes[1].axhline(
        y=1.0,
        color="gray",
        linestyle="--",
        alpha=0.6
    )

    fig.tight_layout()

    # ======================================
    # 7. 保存图片
    # ======================================

    output_file = (
        output_dir / "06_spinel_normalization.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n图片保存成功：", output_file)


if __name__ == "__main__":
    main()
