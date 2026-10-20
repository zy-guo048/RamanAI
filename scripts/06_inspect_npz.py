
from pathlib import Path
import numpy as np

# 1. 定位文件
project_dir = Path(__file__).resolve().parent.parent

file_path = project_dir / "results/06_spectral_matrix.npz"

# 2. 打开 NPZ 文件
with np.load(file_path, allow_pickle=False) as data:

    # 查看保存了哪些数据
    print("===== 文件包含的内容 =====")
    print(data.files)

    # 读取三个主要数组
    X = data["X"]
    raman_shift = data["raman_shift"]
    materials = data["materials"]

    # 3. 检查形状
    print("\n===== 数据形状 =====")
    print("光谱矩阵 X：", X.shape)
    print("波数轴：", raman_shift.shape)
    print("材料标签：", materials.shape)

    # 4. 查看具体内容
    print("\n===== 前 10 个波数 =====")
    print(raman_shift[:10])

    print("\n===== 第一条光谱前 10 个强度值 =====")
    print(X[0, :10])

    print("\n===== 12 条光谱的材料标签 =====")
    print(materials)

    # 5. 查看处理参数
    print("\n===== 分析参数 =====")
    print("SG 窗口：", data["sg_window_length"].item())
    print("AsLS lam：", data["asls_lam"].item())
    print("插值间隔：", data["grid_step"].item())


import pandas as pd

with np.load(file_path, allow_pickle=False) as data:
    X = data["X"]
    raman_shift = data["raman_shift"]
    materials = data["materials"]
    file_names = data["file_names"]

# 将每个拉曼位移作为一列
df = pd.DataFrame(
    X,
    columns=[
        f"{shift:.1f}"
        for shift in raman_shift
    ]
)

# 在表格前面添加样品信息
df.insert(0, "File_Name", file_names)
df.insert(0, "Material", materials)

output_csv = project_dir / "results/06_spectral_matrix.csv"

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

print("CSV 已导出：", output_csv)
