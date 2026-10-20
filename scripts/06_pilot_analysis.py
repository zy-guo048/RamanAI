
"""
RamanAI 实验 6.3

目标：
1. 自动筛选 RAW 光谱
2. 选择前 3 条光谱
3. 逐条进行完整预处理
4. 自动寻峰和特征提取
5. 在终端显示结果
"""

from pathlib import Path
import csv

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    detect_peaks,
    extract_peak_features
)


# ======================================
# 1. 设置数据集路径
# ======================================

project_dir = Path(__file__).resolve().parent.parent

data_dir = project_dir / "fair_oriented"


# ======================================
# 2. 查找 RAW 光谱文件
# ======================================

raw_files = sorted(
    file
    for file in data_dir.rglob("*")
    if file.is_file()
    and file.suffix.lower() == ".txt"
    and "raman_data_raw" in file.name.lower()
)

print("找到 RAW 光谱数量：", len(raw_files))

if not raw_files:
    raise RuntimeError("没有找到 RAW 光谱文件")


# ======================================
# 3. 先选取前 3 条光谱
# ======================================

test_files = raw_files
all_rows = []

print("本次分析数量：", len(test_files))


# ======================================
# 4. 逐条分析光谱
# ======================================

for i, file_path in enumerate(test_files, start=1):

    print("\n" + "=" * 50)
    print(f"正在分析第 {i} 条光谱")
    print("文件名：", file_path.name)

    # 读取原始光谱
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

    # 自动寻峰
    peaks, properties = detect_peaks(
        corrected,
        prominence=1000,
        distance=10
    )

    # 提取当前光谱的峰特征
    features = extract_peak_features(
        corrected,
        peaks,
        properties
    )

    # 先按照两个下划线分割文件名
    name_parts = file_path.name.split("__")

    # 再提取材料名称与样品编号
    material = name_parts[0]
    sample_id = name_parts[1]

    print("材料名称：", material)
    print("样品编号：", sample_id)

    # 把当前光谱的全部候选峰加入汇总列表
    for feature in features:

        row = {
            "Material": material,
            "Sample_ID": sample_id,
            "File_Name": file_path.name,
            "Peak_ID": feature["Peak_ID"],
            "Position_cm-1": feature["Position_cm-1"],
            "Intensity": feature["Intensity"],
            "Prominence": feature["Prominence"],
            "Half_Prominence_Width_cm-1":
                feature["Half_Prominence_Width_cm-1"]
        }

        all_rows.append(row)


    # 输出分析信息
    print(
        "数据点数：",
        len(spectrum.spectral_axis)
    )

    print(
        "检测到候选峰数量：",
        len(features)
    )

    positions = [
        round(feature["Position_cm-1"], 2)
        for feature in features
    ]

    print("候选峰位 (cm^-1)：", positions)


print("\n前三条 RAW 光谱分析完成！")


# ======================================
# 5. 导出多光谱峰特征汇总表
# ======================================

output_dir = project_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

csv_file = output_dir / "06_all_raw_peak_features.csv"

columns = [
    "Material",
    "Sample_ID",
    "File_Name",
    "Peak_ID",
    "Position_cm-1",
    "Intensity",
    "Prominence",
    "Half_Prominence_Width_cm-1"
]

with csv_file.open(
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=columns
    )

    writer.writeheader()
    writer.writerows(all_rows)


print("\n===== CSV 导出结果 =====")
print("光谱数量：", len(test_files))
print("候选峰总数量：", len(all_rows))
print("保存位置：", csv_file)

