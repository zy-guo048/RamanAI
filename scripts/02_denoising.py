
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp

# =========================
# 1. 读取原始拉曼光谱
# =========================

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data/raw/sample_raw.txt"

data = np.loadtxt(
    file_path,
    delimiter=",",
    comments="#"
)

raman_shift = data[:, 0]
intensity = data[:, 1]

spectrum = rp.Spectrum(
    intensity,
    raman_shift
)

print("原始光谱读取成功！")

# =========================
# 2. 使用 SG 滤波去噪
# =========================

denoiser = rp.preprocessing.denoise.SavGol(
    window_length=20,
    polyorder=3
)

denoised = denoiser.apply(spectrum)

print("SG 滤波完成！")

# =========================
# 3. 绘制处理前后的光谱
# =========================

x = spectrum.spectral_axis
y_raw = spectrum.spectral_data
y_smooth = denoised.spectral_data

fig, axes = plt.subplots(
    2, 1, figsize=(11, 8)
)

# 上图：完整光谱对比
axes[0].plot(
    x, y_raw,
    label="Original",
    alpha=0.55
)

axes[0].plot(
    x, y_smooth,
    label="SG Denoised",
    linewidth=1.5
)

axes[0].set_title("Full Raman Spectrum")
axes[0].set_xlabel("Raman Shift (cm^-1)")
axes[0].set_ylabel("Intensity (a.u.)")
axes[0].legend()

# 下图：局部放大
axes[1].plot(
    x, y_raw,
    label="Original",
    alpha=0.55
)

axes[1].plot(
    x, y_smooth,
    label="SG Denoised",
    linewidth=1.5
)

axes[1].set_xlim(950, 1200)
axes[1].set_title("Zoomed Spectrum")
axes[1].set_xlabel("Raman Shift (cm^-1)")
axes[1].set_ylabel("Intensity (a.u.)")
axes[1].legend()

plt.tight_layout()

# =========================
# 4. 保存结果
# =========================

output_dir = project_dir / "results"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "02_denoising.png"

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("图片已保存至：", output_file)
