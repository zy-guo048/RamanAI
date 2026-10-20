
"""
RamanAI 实验 6.12

将两条 Spinel 光谱插值到共同的拉曼位移网格。
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


# 1. 项目路径
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "fair_oriented"
output_dir = project_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. 查找两条 Spinel RAW 光谱
def find_file(sample_id):
    matches = sorted(
        file
        for file in data_dir.rglob("*.txt")
        if file.name.startswith(f"Spinel__{sample_id}__")
        and "__514__45-000__ccw__" in file.name
        and "raman_data_raw" in file.name.lower()
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{sample_id}: 找到 {len(matches)} 个文件，预期为 1"
        )

    return matches[0]


# 3. 预处理及归一化
def prepare_spectrum(file_path):
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

    return normalize_spectrum(corrected)


file_a = find_file("R050184-3")
file_b = find_file("R050259-3")

spectrum_a = prepare_spectrum(file_a)
spectrum_b = prepare_spectrum(file_b)

x_a = spectrum_a.spectral_axis
y_a = spectrum_a.spectral_data

x_b = spectrum_b.spectral_axis
y_b = spectrum_b.spectral_data


# 4. 计算两条光谱共同覆盖的波数范围
common_min = max(x_a.min(), x_b.min())
common_max = min(x_a.max(), x_b.max())

# 网格间隔：每 2 cm^-1 一个点
step = 2.0

start = np.ceil(common_min / step) * step
stop = np.floor(common_max / step) * step

if stop <= start:
    raise ValueError("两条光谱没有足够的共同波数范围")

common_x = np.arange(
    start,
    stop + step / 2,
    step
)


# 5. 线性插值到相同的波数轴
aligned_a = np.interp(
    common_x,
    x_a,
    y_a
)

aligned_b = np.interp(
    common_x,
    x_b,
    y_b
)


# 6. 组成二维光谱矩阵
X = np.vstack([
    aligned_a,
    aligned_b
])

print("光谱 A 原始数据点数：", len(x_a))
print("光谱 B 原始数据点数：", len(x_b))

print(
    "共同波数范围：",
    common_x[0],
    "~",
    common_x[-1],
    "cm^-1"
)

print("共同网格点数：", len(common_x))
print("光谱矩阵形状：", X.shape)

assert X.shape == (2, len(common_x))


# 7. 可视化
fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 8)
)

axes[0].plot(
    x_a, y_a,
    label="Spinel R050184-3"
)

axes[0].plot(
    x_b, y_b,
    label="Spinel R050259-3"
)

axes[0].set_title("Before Spectral Alignment")
axes[0].set_ylabel("Normalized Intensity")
axes[0].legend()
axes[0].grid(alpha=0.2)


axes[1].plot(
    common_x,
    aligned_a,
    label="Aligned R050184-3"
)

axes[1].plot(
    common_x,
    aligned_b,
    label="Aligned R050259-3"
)

axes[1].set_title("After Linear Interpolation")
axes[1].set_xlabel("Raman Shift (cm^-1)")
axes[1].set_ylabel("Normalized Intensity")
axes[1].legend()
axes[1].grid(alpha=0.2)

fig.tight_layout()

output_file = output_dir / "06_aligned_spinel.png"

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("图片已保存：", output_file)
