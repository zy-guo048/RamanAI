
"""
RamanAI 实验 6.7

绘制多材料拉曼峰位分布图。

横轴：Raman Shift (cm^-1)
纵轴：不同的光谱文件
颜色：材料类别
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ======================================
# 1. 设置路径
# ======================================

project_dir = Path(__file__).resolve().parent.parent

input_file = (
    project_dir
    / "results"
    / "06_all_raw_peak_features.csv"
)

output_dir = project_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "06_raman_peak_map.png"


# ======================================
# 2. 读取峰级特征数据
# ======================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig"
)

if df.empty:
    raise ValueError("CSV 文件没有峰记录")

print("成功读取峰级数据！")
print("候选峰数量：", len(df))


# ======================================
# 3. 整理光谱文件清单
# ======================================

spectra = (
    df[["Material", "Sample_ID", "File_Name"]]
    .drop_duplicates()
    .sort_values(["Material", "Sample_ID", "File_Name"])
    .reset_index(drop=True)
)

print("光谱数量：", len(spectra))


# ======================================
# 4. 为不同材料设置颜色
# ======================================

materials = sorted(df["Material"].unique())

cmap = plt.get_cmap("tab10")

material_colors = {
    material: cmap(i)
    for i, material in enumerate(materials)
}

print("材料类别：", materials)


# ======================================
# 5. 创建绘图窗口
# ======================================

fig, ax = plt.subplots(figsize=(13, 8))

y_labels = []

# 避免图例中重复显示同一种材料
plotted_materials = set()


# ======================================
# 6. 逐条光谱绘制峰位
# ======================================

for i, row in spectra.iterrows():

    material = row["Material"]
    sample_id = row["Sample_ID"]
    file_name = row["File_Name"]

    # 找到当前文件的所有候选峰
    group = df[
        df["File_Name"] == file_name
    ]

    # 横轴：候选峰的位置
    x_positions = group["Position_cm-1"]

    # 纵轴：当前光谱的编号
    y_positions = [i] * len(group)

    # 每种材料只添加一次图例
    if material not in plotted_materials:
        legend_label = material
        plotted_materials.add(material)
    else:
        legend_label = "_nolegend_"

    ax.scatter(
        x_positions,
        y_positions,
        color=material_colors[material],
        s=55,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        label=legend_label
    )

    # 从文件名提取测量条件
    parts = file_name.split("__")

    angle = parts[4] if len(parts) > 4 else "unknown"

    condition = (
        parts[5]
        if len(parts) > 5 and parts[5]
        else "unspecified"
    )

    label = (
        f"{material} | {sample_id} "
        f"| {angle} {condition}"
    )

    y_labels.append(label)


# ======================================
# 7. 设置坐标轴
# ======================================

ax.set_yticks(range(len(spectra)))
ax.set_yticklabels(y_labels, fontsize=9)

# 第一条光谱显示在最上面
ax.invert_yaxis()

# 根据数据自动设置横轴范围
min_shift = df["Position_cm-1"].min()
max_shift = df["Position_cm-1"].max()

ax.set_xlim(
    max(0, min_shift - 50),
    max_shift + 50
)

ax.set_xlabel(
    "Raman Shift (cm$^{-1}$)",
    fontsize=12
)

ax.set_ylabel(
    "Raman Spectrum",
    fontsize=12
)

ax.set_title(
    "RamanAI - Multi-Material Raman Peak Map",
    fontsize=14
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    title="Material",
    loc="upper left",
    bbox_to_anchor=(1.01, 1)
)

fig.tight_layout()


# ======================================
# 8. 保存图片
# ======================================

fig.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ======================================
# 9. 输出结果
# ======================================

print("\n===== 绘图完成 =====")
print("材料种类：", len(materials))
print("光谱数量：", len(spectra))
print("绘制候选峰数：", len(df))
print("图片保存位置：", output_file)
