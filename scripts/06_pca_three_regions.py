
"""
RamanAI 实验 6.17

比较三个波数范围的 PCA：
A: 完整范围 232-1466 cm^-1
B: 窄范围   300-850 cm^-1
C: 中间范围 300-1150 cm^-1
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def main():

    # =====================================
    # 1. 设置路径
    # =====================================

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "results"

    input_file = output_dir / "06_spectral_matrix.npz"

    # =====================================
    # 2. 读取光谱矩阵
    # =====================================

    with np.load(input_file, allow_pickle=False) as data:
        X = data["X"]
        raman_shift = data["raman_shift"]
        materials = data["materials"]
        sample_ids = data["sample_ids"]
        file_names = data["file_names"]

    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵")

    if X.shape[1] != len(raman_shift):
        raise ValueError("波数轴长度不匹配")

    if not np.all(np.isfinite(X)):
        raise ValueError("矩阵包含无效数值")

    print("原始光谱矩阵：", X.shape)

    # =====================================
    # 3. 定义三个 PCA 实验组
    # =====================================

    regions = [
        ("Full", "Full: 232-1466", None, None),
        ("Narrow", "Narrow: 300-850", 300, 850),
        ("Middle", "Middle: 300-1150", 300, 1150)
    ]

    colors = {
        "Annite": "tab:blue",
        "Corundum": "tab:orange",
        "Scholzite": "tab:green",
        "Spinel": "tab:red"
    }

    # 每条光谱的基本信息
    results = pd.DataFrame({
        "Spectrum_ID": np.arange(1, len(X) + 1),
        "Material": materials,
        "Sample_ID": sample_ids,
        "File_Name": file_names
    })

    # 创建散点图和权重图
    fig_scores, axes_scores = plt.subplots(
        1, 3, figsize=(18, 6)
    )

    fig_weights, axes_weights = plt.subplots(
        2, 3, figsize=(18, 9)
    )

    # =====================================
    # 4. 依次计算三个 PCA
    # =====================================

    for col, (key, title, low, high) in enumerate(regions):

        # 完整光谱：保留全部列
        if low is None:
            mask = np.ones(len(raman_shift), dtype=bool)

        # 局部光谱：使用布尔索引
        else:
            mask = (
                (raman_shift >= low)
                & (raman_shift <= high)
            )

        region_x = raman_shift[mask]
        region_X = X[:, mask]

        if region_X.shape[1] < 2:
            raise ValueError(f"{key}: 特征数量不足")

        # 每个范围独立计算 PCA
        pca = PCA(n_components=2)
        scores = pca.fit_transform(region_X)

        ratios = pca.explained_variance_ratio_

        print(f"\n===== {key} PCA =====")
        print("数据矩阵形状：", region_X.shape)
        print(f"PC1：{ratios[0]:.2%}")
        print(f"PC2：{ratios[1]:.2%}")
        print(f"合计：{ratios.sum():.2%}")

        # 保存该实验的 PCA 坐标
        results[f"{key}_PC1"] = scores[:, 0]
        results[f"{key}_PC2"] = scores[:, 1]

        # =================================
        # 5. 绘制 PCA 散点图
        # =================================

        ax = axes_scores[col]

        for material in sorted(set(materials)):

            selected = materials == material

            ax.scatter(
                scores[selected, 0],
                scores[selected, 1],
                label=material,
                color=colors[material],
                s=75,
                edgecolors="white",
                linewidths=0.6
            )

        # 标记光谱编号 1-12
        for i in range(len(scores)):
            ax.annotate(
                str(i + 1),
                (scores[i, 0], scores[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({ratios[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ratios[1]:.1%})")

        ax.grid(alpha=0.2)
        ax.legend(title="Material", fontsize=8)

        # =================================
        # 6. 绘制 PC1 和 PC2 权重
        # =================================

        for pc_index in range(2):

            weight_ax = axes_weights[pc_index, col]

            weights = pca.components_[pc_index]

            weight_ax.plot(
                region_x,
                weights,
                color=(
                    "royalblue"
                    if pc_index == 0
                    else "darkorange"
                )
            )

            weight_ax.axhline(
                0,
                color="gray",
                linewidth=0.8,
                alpha=0.5
            )

            # 标出 Scholzite 约1001 cm^-1 的强峰位置
            if region_x.min() <= 1001 <= region_x.max():
                weight_ax.axvline(
                    1001,
                    color="red",
                    linestyle="--",
                    alpha=0.5
                )

            # 在完整范围上标出高波数末端
            if key == "Full":
                weight_ax.axvspan(
                    1350,
                    region_x.max(),
                    color="gray",
                    alpha=0.1
                )

            weight_ax.set_title(
                f"{title} - PC{pc_index + 1} Weights"
            )

            weight_ax.set_xlabel("Raman Shift (cm^-1)")
            weight_ax.set_ylabel("Component Weight")
            weight_ax.grid(alpha=0.2)

            # 打印权重绝对值最大的3个波数
            top_indices = np.argsort(
                np.abs(weights)
            )[-3:][::-1]

            print(
                f"\nPC{pc_index + 1} 权重绝对值"
                "最大的3个波数："
            )

            for index in top_indices:
                print(
                    f"  {region_x[index]:.1f} cm^-1"
                    f" | {weights[index]:.5f}"
                )

    # =====================================
    # 7. 保存 PCA 散点图
    # =====================================

    fig_scores.tight_layout()

    scores_image = (
        output_dir / "06_pca_three_regions.png"
    )

    fig_scores.savefig(
        scores_image,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig_scores)

    # =====================================
    # 8. 保存 PCA 权重图
    # =====================================

    fig_weights.tight_layout()

    weights_image = (
        output_dir / "06_pca_three_regions_weights.png"
    )

    fig_weights.savefig(
        weights_image,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig_weights)

    # =====================================
    # 9. 保存 PCA 坐标 CSV
    # =====================================

    csv_file = (
        output_dir / "06_pca_three_regions_scores.csv"
    )

    results.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== 输出文件 =====")
    print("PCA 散点图：", scores_image)
    print("PCA 权重图：", weights_image)
    print("PCA 坐标：", csv_file)

    print("\n实验 6.17 完成！")


if __name__ == "__main__":
    main()
