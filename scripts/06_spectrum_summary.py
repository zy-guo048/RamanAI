
"""
RamanAI 实验 6.6

目标：
1. 使用 Pandas 读取峰级 CSV
2. 按光谱分组
3. 统计每条光谱的候选峰数量
4. 找到每条光谱中强度最大的候选峰
5. 导出光谱级汇总表
"""

from pathlib import Path
import pandas as pd


# ======================================
# 1. 设置文件路径
# ======================================

project_dir = Path(__file__).resolve().parent.parent

input_file = (
    project_dir
    / "results"
    / "06_all_raw_peak_features.csv"
)

output_file = (
    project_dir
    / "results"
    / "06_spectrum_summary.csv"
)

if not input_file.is_file():
    raise FileNotFoundError(
        f"找不到峰特征文件：{input_file}"
    )


# ======================================
# 2. 读取 CSV
# ======================================

df = pd.read_csv(
    input_file,
    encoding="utf-8-sig"
)

print("CSV 读取成功！")
print("峰记录数量：", len(df))


# ======================================
# 3. 检查需要的字段
# ======================================

required_columns = [
    "Material",
    "Sample_ID",
    "File_Name",
    "Position_cm-1",
    "Intensity"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"CSV 缺少必要字段：{column}"
        )

# 确保峰位和强度是数字
df["Position_cm-1"] = pd.to_numeric(
    df["Position_cm-1"],
    errors="raise"
)

df["Intensity"] = pd.to_numeric(
    df["Intensity"],
    errors="raise"
)


# ======================================
# 4. 按光谱分组
# ======================================

grouped = df.groupby(
    ["Material", "Sample_ID", "File_Name"],
    sort=True
)

summary_rows = []


# ======================================
# 5. 逐组统计
# ======================================

for (material, sample_id, file_name), group in grouped:

    # 当前光谱检测到的候选峰数量
    peak_count = len(group)

    # 查找强度最大的候选峰
    strongest_index = group["Intensity"].idxmax()

    strongest_peak = group.loc[strongest_index]

    # 构建当前光谱的汇总记录
    row = {
        "Material": material,
        "Sample_ID": sample_id,
        "File_Name": file_name,
        "Peak_Count": peak_count,
        "Strongest_Peak_cm-1": round(
            float(strongest_peak["Position_cm-1"]),
            2
        ),
        "Strongest_Peak_Intensity": round(
            float(strongest_peak["Intensity"]),
            2
        )
    }

    summary_rows.append(row)


# ======================================
# 6. 转换为 DataFrame
# ======================================

summary_df = pd.DataFrame(summary_rows)


# ======================================
# 7. 保存光谱级汇总表
# ======================================

summary_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# ======================================
# 8. 输出统计结果
# ======================================

print("\n===== 光谱级数据汇总 =====")

print("候选峰记录数：", len(df))
print("光谱文件数量：", len(summary_df))

print(
    "材料种类数量：",
    summary_df["Material"].nunique()
)

print(
    "不同材料-样品编号组合数：",
    len(
        summary_df[
            ["Material", "Sample_ID"]
        ].drop_duplicates()
    )
)

print("\n===== 每条光谱的汇总信息 =====")

print(
    summary_df[
        [
            "Material",
            "Sample_ID",
            "Peak_Count",
            "Strongest_Peak_cm-1"
        ]
    ].to_string(index=False)
)

print("\n汇总 CSV 已保存：", output_file)