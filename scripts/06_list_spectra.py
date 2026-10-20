
"""
RamanAI 实验 6.1

目标：
1. 自动扫描文件夹
2. 查找所有 TXT 文件
3. 统计光谱文件数量
4. 显示前 10 个文件名
"""

from pathlib import Path


# ======================================
# 1. 获取项目根目录
# ======================================

project_dir = Path(__file__).resolve().parent.parent


# ======================================
# 2. 指定数据集文件夹
# ======================================

data_dir = project_dir / "fair_oriented"


# ======================================
# 3. 检查文件夹是否存在
# ======================================

if not data_dir.is_dir():
    raise FileNotFoundError(
        f"找不到数据文件夹: {data_dir}"
    )


# ======================================
# 4. 查找全部 TXT 文件
# ======================================

txt_files = sorted(
    file
    for file in data_dir.rglob("*")
    if file.is_file() and file.suffix.lower() == ".txt"
)


# ======================================
# 5. 统计文件数量
# ======================================

print("扫描目录：", data_dir)

print("找到的 TXT 文件数量：", len(txt_files))


# ======================================
# 6. 显示前 10 个文件
# ======================================

print("\n===== 前 10 个 TXT 文件 =====")

for i, file in enumerate(txt_files[:10], start=1):

    relative_path = file.relative_to(project_dir)

    print(f"{i}. {relative_path}")


if len(txt_files) == 0:
    print("没有找到 TXT 文件，请检查数据集目录。")
