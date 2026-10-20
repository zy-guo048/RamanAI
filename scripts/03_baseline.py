
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp

# 找到项目根目录
project_dir = Path(__file__).resolve().parent.parent

# 原始光谱文件路径
file_path = project_dir / "data/raw/sample_raw.txt"

# 读取数据
data = np.loadtxt(
    file_path,
    delimiter=",",
    comments="#"
)

# 第一列：拉曼位移
raman_shift = data[:, 0]

# 第二列：拉曼强度
intensity = data[:, 1]

# 创建 RamanSPy 光谱对象
spectrum = rp.Spectrum(intensity, raman_shift)

print("原始光谱读取完成！")
print("数据点数：", len(raman_shift))


# 创建 SG 滤波器
denoiser = rp.preprocessing.denoise.SavGol(
    window_length=9,
    polyorder=3
)

# 将滤波器应用到原始光谱
denoised = denoiser.apply(spectrum)

print("SG 去噪完成！")


# 创建 AsLS 基线校正器
baseline_corrector = rp.preprocessing.baseline.ASLS(
    lam=1e6,
    p=0.01
)

# 对去噪后的光谱进行基线校正
corrected = baseline_corrector.apply(denoised)

print("AsLS 基线校正完成！")


# 获取拉曼位移数组
x = denoised.spectral_axis

# 获取去噪后的光谱强度
y_before = denoised.spectral_data

# 获取基线校正后的强度
y_after = corrected.spectral_data

# 反推出算法估计的基线
estimated_baseline = y_before - y_after

print("基线数据计算完成！")


# 创建上下两张子图
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

# 上图：去噪光谱与估计背景
axes[0].plot(
    x, y_before,
    label="Denoised Spectrum",
    color="blue"
)

axes[0].plot(
    x, estimated_baseline,
    label="Estimated Baseline",
    color="red"
)

axes[0].set_title("Baseline Estimation")
axes[0].set_xlabel("Raman Shift (cm^-1)")
axes[0].set_ylabel("Intensity (a.u.)")
axes[0].legend()

# 下图：基线校正后的光谱
axes[1].plot(
    x, y_after,
    label="Corrected Spectrum",
    color="green"
)

axes[1].axhline(0, color="gray", linestyle="--")

axes[1].set_title("Baseline Corrected Spectrum")
axes[1].set_xlabel("Raman Shift (cm^-1)")
axes[1].set_ylabel("Intensity (a.u.)")
axes[1].legend()

plt.tight_layout()


# 创建结果目录
output_dir = project_dir / "results"
output_dir.mkdir(exist_ok=True)

# 保存图片
output_file = output_dir / "03_baseline.png"

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("实验三完成！")
print("结果保存到：", output_file)