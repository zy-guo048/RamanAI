
"""
RamanAI 实验 6.19

检查实验六的关键数据和分析参数。
不会修改任何原始数据或结果文件。
"""

from pathlib import Path
from collections import Counter

import numpy as np


def main():

    project_dir = Path(__file__).resolve().parent.parent
    results_dir = project_dir / "results"

    matrix_file = results_dir / "06_spectral_matrix.npz"

    # 1. 检查 NPZ 是否存在
    if not matrix_file.is_file():
        raise FileNotFoundError(matrix_file)

    # 2. 读取数据
    with np.load(matrix_file, allow_pickle=False) as data:
        X = data["X"]
        x = data["raman_shift"]
        materials = data["materials"]
        sample_ids = data["sample_ids"]
        file_names = data["file_names"]

        sg_window = int(data["sg_window_length"])
        sg_polyorder = int(data["sg_polyorder"])
        asls_lam = float(data["asls_lam"])
        asls_p = float(data["asls_p"])
        grid_step = float(data["grid_step"])

    # 3. 验证矩阵结构
    if X.shape != (12, 618):
        raise ValueError(f"矩阵形状不正确：{X.shape}")

    if not (
        len(x) == len(X[0])
        and len(materials) == len(sample_ids) == len(file_names) == 12
    ):
        raise ValueError("数据与标签长度不一致")

    if not np.all(np.isfinite(X)):
        raise ValueError("矩阵包含 NaN 或 Inf")

    # 4. 检查波数轴
    np.testing.assert_allclose(
        [x[0], x[-1]],
        [232.0, 1466.0]
    )

    np.testing.assert_allclose(
        np.diff(x),
        2.0
    )

    # 5. 检查标签
    counts = Counter(materials.tolist())

    expected = {
        "Annite": 2,
        "Corundum": 4,
        "Scholzite": 1,
        "Spinel": 5
    }

    if counts != expected:
        raise ValueError(f"材料标签不符合预期：{counts}")

    if len(set(sample_ids.tolist())) != 6:
        raise ValueError("独立样品编号数量不符合预期")

    # 6. 检查预处理参数
    if not (
        sg_window == 21
        and sg_polyorder == 3
        and np.isclose(asls_lam, 1e6)
        and np.isclose(asls_p, 0.01)
        and np.isclose(grid_step, 2.0)
    ):
        raise ValueError("预处理参数与实验记录不一致")

    # 7. 检查重要结果文件
    required_files = [
        "06_spectral_heatmap.png",
        "06_pca_scores.png",
        "06_pca_three_regions.png",
        "06_pca_reconstruction.png",
        "06_pca_reconstruction_errors.csv"
    ]

    for filename in required_files:
        path = results_dir / filename

        if not path.is_file():
            raise FileNotFoundError(path)

    print("===== RamanAI 实验六检查 =====")
    print("光谱矩阵：", X.shape)
    print("波数范围：", x[0], "~", x[-1])
    print("不同样品编号：", len(set(sample_ids.tolist())))
    print("材料分布：", dict(counts))
    print("SG窗口：", sg_window)
    print("AsLS lam：", asls_lam)
    print("AsLS p：", asls_p)
    print("网格间隔：", grid_step)

    print("\n全部检查通过！")
    print("实验六数据已准备好用于 V1 开发。")


if __name__ == "__main__":
    main()
