
"""
RamanAI 实验 6.18

目标：
1. 使用两个主成分进行 PCA 降维
2. 利用 inverse_transform 重建光谱
3. 计算重建误差
4. 比较真实 Spinel 光谱与重建光谱
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
    # 2. 读取光谱矩阵
    # ======================================

    with np.load(input_file, allow_pickle=False) as data:
        X = data["X"]
        raman_shift = data["raman_shift"]
        materials = data["materials"]
        sample_ids = data["sample_ids"]
        file_names = data["file_names"]

    if X.ndim != 2:
        raise ValueError("光谱矩阵必须是二维数组")

    if X.shape[1] != len(raman_shift):
        raise ValueError("光谱矩阵与波数轴不匹配")

    if not np.all(np.isfinite(X)):
        raise ValueError("光谱包含无效数值")

    print("原始光谱矩阵：", X.shape)

    # ======================================
    # 3. PCA 降维
    # ======================================

    pca = PCA(n_components=2)

    scores = pca.fit_transform(X)

    print("PCA 降维后矩阵：", scores.shape)

    # ======================================
    # 4. PCA 逆变换
    # ======================================

    X_reconstructed = pca.inverse_transform(
        scores
    )

    print(
        "PCA 重建后矩阵：",
        X_reconstructed.shape
    )

    assert X_reconstructed.shape == X.shape

    # ======================================
    # 5. 计算重建误差
    # ======================================

    residual = X - X_reconstructed

    # 每条光谱的均方根误差
    rmse_each = np.sqrt(
        np.mean(residual ** 2, axis=1)
    )

    # 全部光谱的均方根误差
    rmse_all = np.sqrt(
        np.mean(residual ** 2)
    )

    # PCA 解释方差比例
    explained_ratio = (
        pca.explained_variance_ratio_.sum()
    )

    # 相对于每个波数列均值的总平方变化
    centered_X = X - X.mean(axis=0)

    total_variation = np.sum(
        centered_X ** 2
    )

    residual_variation = np.sum(
        residual ** 2
    )

    if total_variation <= 0:
        raise ValueError("数据没有变化，无法计算方差比例")

    retained_ratio = (
        1 - residual_variation / total_variation
    )

    # 验证 PCA 的方差解释与重建误差一致
    np.testing.assert_allclose(
        retained_ratio,
        explained_ratio,
        rtol=1e-8,
        atol=1e-10
    )

    print("\n===== PCA 重建结果 =====")
    print(f"解释方差比例：{explained_ratio:.2%}")
    print(f"重建保留的方差比例：{retained_ratio:.2%}")
    print(f"未保留的方差比例：{1-retained_ratio:.2%}")
    print(f"整体 RMSE：{rmse_all:.6f}")
    print("方差比例验证通过：True")

    # ======================================
    # 6. 选择一条真实 Spinel 光谱
    # ======================================

    matches = np.flatnonzero(
        (materials == "Spinel")
        & (sample_ids == "R050259-3")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"预期找到1条目标光谱，实际找到{len(matches)}条"
        )

    index = int(matches[0])

    original_y = X[index]
    reconstructed_y = X_reconstructed[index]
    error_y = residual[index]

    print("\n===== 目标 Spinel 光谱 =====")
    print("光谱编号：", index + 1)
    print("样品编号：", sample_ids[index])
    print(f"该光谱 RMSE：{rmse_each[index]:.6f}")

    # ======================================
    # 7. 绘制原始与重建光谱
    # ======================================

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True
    )

    axes[0].plot(
        raman_shift,
        original_y,
        label="Original Spectrum",
        color="royalblue",
        linewidth=1.6
    )

    axes[0].plot(
        raman_shift,
        reconstructed_y,
        label="PCA Reconstruction (2 PCs)",
        color="darkorange",
        linestyle="--",
        linewidth=1.6
    )

    axes[0].set_title(
        "RamanAI - PCA Reconstruction of Spinel"
    )

    axes[0].set_ylabel("Normalized Intensity")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # 下图：重建误差
    axes[1].plot(
        raman_shift,
        error_y,
        color="firebrick"
    )

    axes[1].axhline(
        0,
        color="gray",
        linestyle="--",
        alpha=0.6
    )

    axes[1].set_title(
        "Reconstruction Residual (Original - Reconstructed)"
    )

    axes[1].set_xlabel("Raman Shift (cm^-1)")
    axes[1].set_ylabel("Residual Intensity")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()

    image_file = (
        output_dir / "06_pca_reconstruction.png"
    )

    fig.savefig(
        image_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # 8. 保存每条光谱的重建误差
    # ======================================

    results = pd.DataFrame({
        "Spectrum_ID": np.arange(1, len(X) + 1),
        "Material": materials,
        "Sample_ID": sample_ids,
        "File_Name": file_names,
        "RMSE": rmse_each
    })

    csv_file = (
        output_dir / "06_pca_reconstruction_errors.csv"
    )

    results.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== 每条光谱的重建 RMSE =====")
    print(
        results[
            ["Spectrum_ID", "Material", "Sample_ID", "RMSE"]
        ].round(6).to_string(index=False)
    )

    print("\n图片：", image_file)
    print("误差数据：", csv_file)

    print("\n实验 6.18 完成！")


if __name__ == "__main__":
    main()
