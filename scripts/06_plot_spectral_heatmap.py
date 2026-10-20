
"""
RamanAI 实验 6.14

从统一光谱矩阵 NPZ 中读取数据，
绘制12条拉曼光谱的强度热图。
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ======================================
# 1. 设置路径
# ======================================

project_dir = Path(__file__).resolve().parent.parent

input_file = (
    project_dir / "results/06_spectral_matrix.npz"
)

output_file = (
    project_dir / "results/06_spectral_heatmap.png"
)


# ======================================
# 2. 读取 NPZ 文件
# ======================================

with np.load(input_file, allow_pickle=False) as data:

    X = data["X"]
    raman_shift = data["raman_shift"]
    materials = data["materials"]
    sample_ids = data["sample_ids"]
    file_names = data["file_names"]


# ======================================
# 3. 检查数据
# ======================================

if X.ndim != 2:
    raise ValueError("光谱矩阵必须是二维数组")

if X.shape[1] != len(raman_shift):
    raise ValueError("矩阵列数与波数轴不一致")

if X.shape[0] != len(materials):
    raise ValueError("矩阵行数与材料标签不一致")

if not np.all(np.isfinite(X)):
    raise ValueError("光谱矩阵包含无效数值")

print("光谱矩阵形状：", X.shape)

print(
    "拉曼位移范围：",
    raman_shift[0],
    "~",
    raman_shift[-1]
)

print("材料种类：", sorted(set(materials)))


# ======================================
# 4. 构建每行的光谱标签
# ======================================

y_labels = []

for i in range(len(materials)):

    parts = file_names[i].split("__")

    angle = parts[4] if len(parts) > 4 else "unknown"

    condition = (
        parts[5]
        if len(parts) > 5 and parts[5]
        else "unspecified"
    )

    label = (
        f"{materials[i]} | {sample_ids[i]} "
        f"| {angle} {condition}"
    )

    y_labels.append(label)


# ======================================
# 5. 绘制热图
# ======================================

fig, ax = plt.subplots(figsize=(14, 8))

image = ax.imshow(
    X,
    aspect="auto",
    cmap="viridis",
    interpolation="nearest",
    origin="upper"
)


# ======================================
# 6. 设置横轴为真实拉曼位移
# ======================================

tick_indices = np.linspace(
    0,
    len(raman_shift) - 1,
    8,
    dtype=int
)

ax.set_xticks(tick_indices)

ax.set_xticklabels([
    f"{raman_shift[i]:.0f}"
    for i in tick_indices
])


# ======================================
# 7. 设置纵轴标签
# ======================================

ax.set_yticks(np.arange(X.shape[0]))
ax.set_yticklabels(y_labels, fontsize=9)


# ======================================
# 8. 用分隔线区分不同材料
# ======================================

for i in range(1, len(materials)):

    if materials[i] != materials[i - 1]:

        ax.axhline(
            y=i - 0.5,
            color="white",
            linewidth=1.5
        )


# ======================================
# 9. 设置标题和颜色条
# ======================================

ax.set_title(
    "RamanAI - Multi-Material Spectral Heatmap",
    fontsize=14
)

ax.set_xlabel("Raman Shift (cm^-1)")
ax.set_ylabel("Raman Spectra")

colorbar = fig.colorbar(
    image,
    ax=ax,
    pad=0.02
)

colorbar.set_label("Normalized Intensity")


# ======================================
# 10. 保存图片
# ======================================

fig.tight_layout()

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\n热图保存成功：", output_file)
print("光谱数量：", X.shape[0])
print("每条光谱数据点：", X.shape[1])
print("归一化强度范围：", X.min(), "~", X.max())
