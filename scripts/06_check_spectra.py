
"""
RamanAI 实验 6.2

批量测试 RRUFF 光谱文件能否成功读取。
不修改原始数据。
"""

from pathlib import Path
from raman_utils import load_spectrum


# 1. 设置项目和数据路径
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "fair_oriented"


# 2. 收集 TXT 文件
txt_files = sorted(
    file
    for file in data_dir.rglob("*")
    if file.is_file() and file.suffix.lower() == ".txt"
)

print("找到 TXT 文件：", len(txt_files))


# 3. 创建两个列表
successful_files = []
failed_files = []


# 4. 逐个尝试读取文件
for file_path in txt_files:

    try:
        spectrum = load_spectrum(file_path)

        # 记录读取成功的文件
        successful_files.append({
            "file": file_path.name,
            "points": len(spectrum.spectral_axis),
            "min_shift": float(spectrum.spectral_axis.min()),
            "max_shift": float(spectrum.spectral_axis.max())
        })

    except Exception as error:

        # 记录错误，但不中断批量测试
        failed_files.append({
            "file": file_path.name,
            "error": str(error)
        })


# 5. 输出统计结果
print("\n===== 批量读取结果 =====")

print("文件总数：", len(txt_files))
print("读取成功：", len(successful_files))
print("读取失败：", len(failed_files))


# 6. 显示前5个读取成功的文件
print("\n===== 成功示例（前5个）=====")

for item in successful_files[:5]:

    print("\n文件：", item["file"])
    print("数据点数：", item["points"])

    print(
        "波数范围：",
        item["min_shift"],
        "~",
        item["max_shift"]
    )


# 7. 显示读取失败的文件
print("\n===== 读取失败的文件 =====")

for item in failed_files:

    print("\n文件：", item["file"])
    print("错误：", item["error"])


if len(failed_files) == 0:
    print("\n所有 TXT 文件均成功读取！")
