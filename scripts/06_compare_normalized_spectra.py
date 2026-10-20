
"""
RamanAI 实验 6.9

比较两条不同 Spinel 样品的拉曼光谱：
1. 读取两条 RAW 光谱
2. 使用相同的 SG + AsLS 参数预处理
3. 分别进行最大值归一化
4. 绘制归一化前后对比图
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    normalize_spectrum
)


# ======================================
# 1. 设置路径
# ======================================

project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "fair_oriented"

output_dir = project_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)


# ======================================
# 2. 查找指定的 Spinel 光谱
# ======================================

def find_spinel_file(sample_id):

    matches = sorted(
        file
        for file in data_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() == ".txt"
        and file.name.startswith(
            f"Spinel__{sample_id}__"
        )
        and "__45-000__ccw__" in file.name
        and "raman_data_raw" in file.name.lower()
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{sample_id}: 预期找到1个文件，"
            f"实际找到 {len(matches)} 个"
        )

    return matches[0]


file_a = find_spinel_file("R050184-3")
file_b = find_spinel_file("R050259-3")

print("光谱 A：", file_a.name)
print("光谱 B：", file_b.name)


# ======================================
# 3. 定义预处理与归一化函数
# ======================================

def process_and_normalize(file_path):

    # 读取光谱
    spectrum = load_spectrum(file_path)

    # SG 去噪
    denoised = denoise_spectrum(
        spectrum,
        window_length=21,
        polyorder=3
    )

    # AsLS 基线校正
    corrected = baseline_correct_spectrum(
        denoised,
        lam=1e6,
        p=0.01
    )

    # 提取横纵坐标
    x = corrected.spectral_axis
    y = corrected.spectral_data


    # 使用 RamanAI 自定义归一化函数
    normalized = normalize_spectrum(
        corrected,
        method="max"
    )

    normalized_y = normalized.spectral_data

    # 记录归一化前的最大强度
    max_intensity = np.max(y)

    # 验证新旧计算方式一致
    old_normalized_y = y / max_intensity

    np.testing.assert_allclose(
        normalized_y,
        old_normalized_y,
        rtol=0,
        atol=1e-12
    )

    # 保持原函数的四个返回值
    return x, y, normalized_y, max_intensity


    return x, y, normalized_y, max_intensity


# ======================================
# 4. 分别处理两条光谱
# ======================================

x_a, y_a, norm_a, max_a = process_and_normalize(
    file_a
)

x_b, y_b, norm_b, max_b = process_and_normalize(
    file_b
)


# ======================================
# 5. 输出归一化结果
# ======================================

print("\n===== 归一化结果 =====")

print("光谱 A 最大强度：", max_a)
print("光谱 B 最大强度：", max_b)

print("两条光谱最大强度之比 B/A：", max_b / max_a)

print("光谱 A 归一化最大值：", np.max(norm_a))
print("光谱 B 归一化最大值：", np.max(norm_b))

print(
    "A 归一化检查：",
    np.isclose(np.max(norm_a), 1.0)
)

print(
    "B 归一化检查：",
    np.isclose(np.max(norm_b), 1.0)
)


# ======================================
# 6. 绘制对比图
# ======================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 9),
    sharex=True
)

# 上图：归一化前
axes[0].plot(
    x_a,
    y_a,
    label="Spinel R050184-3",
    color="royalblue"
)

axes[0].plot(
    x_b,
    y_b,
    label="Spinel R050259-3",
    color="darkorange"
)

axes[0].set_title(
    "Spinel Spectra - Before Normalization"
)

axes[0].set_ylabel(
    "Corrected Intensity (a.u.)"
)

axes[0].axhline(
    0,
    color="gray",
    linewidth=0.8,
    alpha=0.5
)

axes[0].legend()
axes[0].grid(alpha=0.25)


# 下图：归一化后
axes[1].plot(
    x_a,
    norm_a,
    label="Spinel R050184-3",
    color="royalblue"
)

axes[1].plot(
    x_b,
    norm_b,
    label="Spinel R050259-3",
    color="darkorange"
)

axes[1].set_title(
    "Spinel Spectra - After Max Normalization"
)

axes[1].set_xlabel(
    "Raman Shift (cm^-1)"
)

axes[1].set_ylabel(
    "Normalized Intensity"
)

axes[1].axhline(
    0,
    color="gray",
    linewidth=0.8,
    alpha=0.5
)

axes[1].axhline(
    1,
    color="gray",
    linestyle="--",
    alpha=0.5
)

axes[1].legend()
axes[1].grid(alpha=0.25)


# ======================================
# 7. 保存图片
# ======================================

fig.tight_layout()

output_file = (
    output_dir
    / "06_spinel_comparison_refactored.png"
)

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\n图片保存成功：", output_file)