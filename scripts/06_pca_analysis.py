
"""
RamanAI 实验 6.15

使用 PCA 对 12 条拉曼光谱进行二维降维。

输入：
    results/06_spectral_matrix.npz

输出：
    results/06_pca_scores.png
    results/06_pca_loadings.png
    results/06_pca_scores.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def main():

    # ======================================
    # 1. 设置路径
    # ======================================

    project_dir = Path(__file__).resolve().parent.parent

    input_file = (
        project_dir / "results/06_spectral_matrix.npz"
    )

    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ======================================
    # 2. 读取统一光谱矩阵
    # ======================================

    with np.load(input_file, allow_pickle=False) as data:

        X = data["X"]
        raman_shift = data["raman_shift"]
        materials = data["materials"]
        sample_ids = data["sample_ids"]
        file_names = data["file_names"]

    print("===== 原始数据 =====")
    print("光谱矩阵形状：", X.shape)
    print("材料类别：", sorted(set(materials)))

    # ======================================
    # 3. 验证数据结构
    # ======================================

    if X.ndim != 2:
        raise ValueError("X 必须是二维数组")

    if X.shape[0] < 3:
        raise ValueError("PCA 实验至少需要 3 条光谱")

    if X.shape[1] != len(raman_shift):
        raise ValueError("波数轴长度不匹配")

    if len(materials) != X.shape[0]:
        raise ValueError("材料标签数量不匹配")

    if not np.all(np.isfinite(X)):
        raise ValueError("X 中包含 NaN 或 Inf")

    # ======================================
    # 4. 创建 PCA 模型
    # ======================================

    pca = PCA(n_components=2)

    # 将 618 维光谱转换成二维坐标
    scores = pca.fit_transform(X)

    print("\n===== PCA 结果 =====")
    print("降维后矩阵形状：", scores.shape)

    # ======================================
    # 5. 查看解释方差比例
    # ======================================

    ratios = pca.explained_variance_ratio_

    print(f"PC1 解释方差比例：{ratios[0]:.2%}")
    print(f"PC2 解释方差比例：{ratios[1]:.2%}")
    print(f"前两个主成分合计：{ratios.sum():.2%}")

    # ======================================
    # 6. 绘制 PCA 散点图
    # ======================================

    fig, ax = plt.subplots(figsize=(10, 7))

    unique_materials = sorted(set(materials))

    cmap = plt.get_cmap("tab10")

    colors = {
        material: cmap(i)
        for i, material in enumerate(unique_materials)
    }

    for material in unique_materials:

        mask = materials == material

        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            label=material,
            color=colors[material],
            s=85,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8
        )

    # 为每个点标注编号 1~12
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
        color="gray",
        linestyle="--",
        alpha=0.3
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        alpha=0.3
    )

    ax.set_xlabel(
        f"PC1 ({ratios[0]:.1%} variance)"
    )

    ax.set_ylabel(
        f"PC2 ({ratios[1]:.1%} variance)"
    )

    ax.set_title(
        "RamanAI - PCA of Raman Spectra"
    )

    ax.legend(title="Material")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    scores_image = output_dir / "06_pca_scores.png"

    fig.savefig(
        scores_image,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # 7. 保存 PCA 坐标
    # ======================================

    results = pd.DataFrame({
        "Spectrum_ID": np.arange(1, len(scores) + 1),
        "Material": materials,
        "Sample_ID": sample_ids,
        "File_Name": file_names,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1]
    })

    scores_csv = output_dir / "06_pca_scores.csv"

    results.to_csv(
        scores_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== 每条光谱的 PCA 坐标 =====")

    print(
        results[
            ["Spectrum_ID", "Material", "Sample_ID", "PC1", "PC2"]
        ].round(4).to_string(index=False)
    )

    # ======================================
    # 8. 绘制主成分系数曲线
    # ======================================

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True
    )

    axes[0].plot(
        raman_shift,
        pca.components_[0],
        color="royalblue"
    )

    axes[0].set_title("PC1 Component Weights")
    axes[0].set_ylabel("PC1 Weight")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        raman_shift,
        pca.components_[1],
        color="darkorange"
    )

    axes[1].set_title("PC2 Component Weights")
    axes[1].set_xlabel("Raman Shift (cm^-1)")
    axes[1].set_ylabel("PC2 Weight")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()

    loadings_image = output_dir / "06_pca_loadings.png"

    fig.savefig(
        loadings_image,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # 9. 输出主成分权重最大的波数
    # ======================================

    for i in range(2):

        weights = pca.components_[i]

        top_indices = np.argsort(
            np.abs(weights)
        )[-5:][::-1]

        print(f"\nPC{i + 1} 权重绝对值最大的 5 个波数：")

        for index in top_indices:

            print(
                f"  {raman_shift[index]:.1f} cm^-1"
                f" | 权重：{weights[index]:.5f}"
            )

    # ======================================
    # 10. 完成
    # ======================================

    print("\nPCA 散点图：", scores_image)
    print("主成分权重图：", loadings_image)
    print("PCA 坐标 CSV：", scores_csv)

    print("\n实验 6.15 完成！")


if __name__ == "__main__":
    main()
