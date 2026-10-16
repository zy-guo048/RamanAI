
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ramanspy as rp
import csv

# 引入自动寻峰函数
from scipy.signal import find_peaks, peak_widths

# 1. 获取项目路径
project_dir = Path(__file__).resolve().parent.parent

# 2. 定位数据文件
file_path = project_dir / "data/raw/sample_raw.txt"

# 3. 读取原始数据
data = np.loadtxt(
    file_path,
    delimiter=",",
    comments="#"
)

# 4. 提取波数和强度
raman_shift = data[:, 0]
intensity = data[:, 1]

# 5. 创建 RamanSPy 光谱对象
spectrum = rp.Spectrum(
    intensity,
    raman_shift
)

print("原始光谱读取成功！")


# =========================
# 2. SG 滤波去噪
# =========================

denoiser = rp.preprocessing.denoise.SavGol(
    window_length=21,
    polyorder=3
)

denoised = denoiser.apply(spectrum)

# =========================
# 3. AsLS 基线校正
# =========================

baseline_corrector = rp.preprocessing.baseline.ASLS(
    lam=1e6,
    p=0.01
)

corrected = baseline_corrector.apply(denoised)

# 提取处理后的数据
x = corrected.spectral_axis
y = corrected.spectral_data

print("光谱预处理完成！")


# =========================
# 4. 自动寻找特征峰
# =========================

peaks, properties = find_peaks(
    y,
    prominence=1000,
    distance=10
)

# 输出峰数量
print("检测到的峰数量：", len(peaks))

# 输出每个峰的位置和强度
for i, index in enumerate(peaks):

    peak_position = x[index]
    peak_intensity = y[index]
    peak_prominence = properties["prominences"][i]

    print(
        f"峰 {i+1}: "
        f"位置={peak_position:.2f} cm^-1, "
        f"强度={peak_intensity:.2f}, "
        f"突出度={peak_prominence:.2f}"
    )


# =========================
# 5. 绘制特征峰
# =========================

fig, ax = plt.subplots(figsize=(11, 5))

# 绘制拉曼光谱
ax.plot(
    x,
    y,
    color="blue",
    label="Corrected Raman Spectrum"
)

# 用红点标记检测到的峰
ax.scatter(
    x[peaks],
    y[peaks],
    color="red",
    s=45,
    label="Detected Peaks",
    zorder=3
)

# 设置标题和坐标
ax.set_title("Raman Peak Detection")
ax.set_xlabel("Raman Shift (cm^-1)")
ax.set_ylabel("Intensity (a.u.)")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()


# =========================
# 6. 保存实验结果
# =========================

output_dir = project_dir / "results"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "04_peaks.png"

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("实验四完成！")
print("图片保存位置：", output_file)


# =========================
# 比较不同 prominence 阈值
# =========================

thresholds = [200, 500, 1000]

for threshold in thresholds:

    test_peaks, test_properties = find_peaks(
        y,
        prominence=threshold,
        distance=10
    )

    print(f"\nprominence = {threshold}")
    print(f"检测到 {len(test_peaks)} 个峰")

    print("峰位：")
    print(np.round(x[test_peaks], 2))


# =========================
# 7. 计算候选峰的宽度
# =========================

# 固定阈值，重新获取8个主要候选峰
peaks, properties = find_peaks(
    y,
    prominence=1000,
    distance=10
)

# 在半突出度处计算宽度
width_results = peak_widths(
    y,
    peaks,
    rel_height=0.5
)

# 返回的宽度以采样点为单位
width_samples = width_results[0]

# 左右交点的浮点索引
left_ips = width_results[2]
right_ips = width_results[3]

# 将采样点索引转换为真实拉曼位移
indices = np.arange(len(x))

left_positions = np.interp(
    left_ips, indices, x
)

right_positions = np.interp(
    right_ips, indices, x
)

# 宽度单位转换为 cm^-1
width_cm = right_positions - left_positions

# 输出结果
print("\n候选峰宽度分析：")

for i, peak in enumerate(peaks):
    print(
        f"峰位: {x[peak]:.2f} cm^-1, "
        f"半突出度宽度: {width_cm[i]:.2f} cm^-1"
    )


# =========================
# 8. 特征峰宽度可视化
# =========================

fig, ax = plt.subplots(figsize=(12, 6))

# 绘制基线校正后的光谱
ax.plot(
    x,
    y,
    color="blue",
    label="Corrected Raman Spectrum"
)

# 标记峰顶位置
ax.scatter(
    x[peaks],
    y[peaks],
    color="red",
    s=45,
    zorder=3,
    label="Detected Peaks"
)

# 绘制半突出度宽度
ax.hlines(
    y=width_results[1],
    xmin=left_positions,
    xmax=right_positions,
    color="orange",
    linewidth=3,
    label="Peak Width at Half Prominence"
)

# 标记每个峰的宽度数值
for i, peak in enumerate(peaks):

    ax.annotate(
        f"{width_cm[i]:.1f}",
        xy=(x[peak], y[peak]),
        xytext=(5, 12),
        textcoords="offset points",
        fontsize=9,
        color="darkred"
    )

# 设置坐标轴和标题
ax.set_title("Raman Peak Width Analysis")
ax.set_xlabel("Raman Shift (cm^-1)")
ax.set_ylabel("Intensity (a.u.)")

ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()

# 保存图片
output_dir = project_dir / "results"
output_dir.mkdir(exist_ok=True)

fig.savefig(
    output_dir / "04_peak_widths.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("峰宽可视化完成！")


# =========================
# 9. 保存拉曼峰特征到 CSV
# =========================

# 设置保存路径
csv_file = output_dir / "04_peak_features.csv"

# 打开 CSV 文件
with open(
    csv_file,
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    # 写入表头
    writer.writerow([
        "Peak_ID",
        "Position_cm-1",
        "Intensity",
        "Prominence",
        "Half_Prominence_Width_cm-1"
    ])

    # 逐个保存检测到的峰
    for i, peak in enumerate(peaks):

        position = x[peak]
        peak_intensity = y[peak]
        prominence = properties["prominences"][i]
        width = width_cm[i]

        writer.writerow([
            i + 1,
            round(float(position), 2),
            round(float(peak_intensity), 2),
            round(float(prominence), 2),
            round(float(width), 2)
        ])

print("拉曼峰特征保存成功！")
print("CSV 文件位置：", csv_file)


