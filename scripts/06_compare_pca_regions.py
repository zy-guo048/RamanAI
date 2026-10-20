
"""
RamanAI 实验 6.16

比较两种波数范围的 PCA：

A: 完整光谱 232-1466 cm^-1
B: 局部光谱 300-850 cm^-1
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def main():

    # ====================================
    # 1. 路径
    # ====================================

    project_dir = Path(__file__).resolve().parent.parent

    input_file = (
        project_dir / "results/06_spectral_matrix.npz"
    )

    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ====================================
    # 2. 读取光谱矩阵
    # ====================================

    with np.load(input_file, allow_pickle=False) as data:

        X = data["X"]
        raman_shift = data["raman_shift"]
        materials = data["materials"]
        sample_ids = data["sample_ids"]

    if X.ndim != 2 or X.shape[1] != len(raman_shift):
        raise ValueError("光谱矩阵和波数轴不匹配")

    if not np.all(np.isfinite(X)):
        raise ValueError("光谱矩阵存在无效数值")

    print("===== 原始数据 =====")
    print("完整矩阵形状：", X.shape)

    # ====================================
    # 3. 选择 300-850 cm^-1 波数范围
    # ====================================

    min_shift = 300
    max_shift = 850

    mask = (
        (raman_shift >= min_shift)
        & (raman_shift <= max_shift)
    )

    X_region = X[:, mask]

    region_axis = raman_shift[mask]

    if X_region.shape[1] < 2:
        raise ValueError("所选区域的数据点不足")

    print("\n===== 局部光谱 =====")
    print("局部矩阵形状：", X_region.shape)
    print(
        "实际波数范围：",
        region_axis[0],
        "~",
        region_axis[-1]
    )

    # ====================================
    # 4. 完整光谱 PCA
    # ====================================

    pca_full = PCA(n_components=2)

    scores_full = pca_full.fit_transform(X)

    # ====================================
    # 5. 局部光谱 PCA
    # ====================================

    pca_region = PCA(n_components=2)

    scores_region = pca_region.fit_transform(
        X_region
    )

    # ====================================
    # 6. 比较解释方差比例
    # ====================================

    full_ratio = pca_full.explained_variance_ratio_
    region_ratio = pca_region.explained_variance_ratio_

    print("\n===== PCA 对比结果 =====")

    print("完整光谱：")
    print(f"  PC1：{full_ratio[0]:.2%}")
    print(f"  PC2：{full_ratio[1]:.2%}")
    print(f"  合计：{full_ratio.sum():.2%}")

    print("\n局部光谱：")
    print(f"  PC1：{region_ratio[0]:.2%}")
    print(f"  PC2：{region_ratio[1]:.2%}")
    print(f"  合计：{region_ratio.sum():.2%}")

    # ====================================
    # 7. 并排绘制 PCA 散点图
    # ====================================

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 6)
    )

    colors = {
        "Annite": "tab:blue",
        "Corundum": "tab:orange",
        "Scholzite": "tab:green",
        "Spinel": "tab:red"
    }

    experiments = [
        (
            axes[0],
            scores_full,
            full_ratio,
            "Full Spectrum PCA (232-1466 cm^-1)"
        ),
        (
            axes[1],
            scores_region,
            region_ratio,
            "Region PCA (300-850 cm^-1)"
        )
    ]

    for ax, scores, ratios, title in experiments:

        for material in sorted(set(materials)):

            selected = materials == material

            ax.scatter(
                scores[selected, 0],
                scores[selected, 1],
                label=material,
                color=colors[material],
                s=75,
                alpha=0.85
            )

        # 标记对应光谱的编号
        for i in range(len(scores)):

            ax.annotate(
                str(i + 1),
                (scores[i, 0], scores[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            alpha=0.3
        )

        ax.axvline(
            0,
            linestyle="--",
            color="gray",
            alpha=0.3
        )

        ax.set_title(title)

        ax.set_xlabel(
            f"PC1 ({ratios[0]:.1%})"
        )

        ax.set_ylabel(
            f"PC2 ({ratios[1]:.1%})"
        )

        ax.grid(alpha=0.2)
        ax.legend(title="Material")

    fig.tight_layout()

    image_file = (
        output_dir / "06_pca_region_comparison.png"
    )

    fig.savefig(
        image_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ====================================
    # 8. 保存两组 PCA 坐标
    # ====================================

    results = pd.DataFrame({
        "Spectrum_ID": np.arange(1, len(X) + 1),
        "Material": materials,
        "Sample_ID": sample_ids,
        "Full_PC1": scores_full[:, 0],
        "Full_PC2": scores_full[:, 1],
        "Region_PC1": scores_region[:, 0],
        "Region_PC2": scores_region[:, 1]
    })

    csv_file = (
        output_dir / "06_pca_region_comparison.csv"
    )

    results.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ====================================
    # 9. 查看局部 PCA 的重要波数
    # ====================================

    for pc_index in range(2):

        weights = pca_region.components_[pc_index]

        top_indices = np.argsort(
            np.abs(weights)
        )[-5:][::-1]

        print(
            f"\n局部 PCA - PC{pc_index + 1}"
            " 权重绝对值最大的5个波数："
        )

        for index in top_indices:

            print(
                f"  {region_axis[index]:.1f} cm^-1"
                f" | 权重：{weights[index]:.5f}"
            )

    print("\n对比图：", image_file)
    print("对比数据：", csv_file)

    print("\n实验 6.16 完成！")


if __name__ == "__main__":
    main()
