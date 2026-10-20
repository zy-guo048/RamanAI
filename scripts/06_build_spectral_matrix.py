
"""
RamanAI 实验 6.13

目标：
1. 批量读取全部 RAW 拉曼光谱
2. 统一预处理和归一化
3. 建立共同波数网格
4. 使用线性插值构建二维矩阵
5. 保存矩阵、标签和光谱文件信息
"""

from pathlib import Path

import numpy as np

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    normalize_spectrum
)


def main():

    # ======================================
    # 1. 设置路径和参数
    # ======================================

    project_dir = Path(__file__).resolve().parent.parent

    data_dir = project_dir / "fair_oriented"

    output_dir = project_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "06_spectral_matrix.npz"
    )

    SG_WINDOW = 21
    SG_POLYORDER = 3

    ASLS_LAM = 1e6
    ASLS_P = 0.01

    GRID_STEP = 2.0

    # ======================================
    # 2. 查找全部 RAW 文件
    # ======================================

    raw_files = sorted(
        file
        for file in data_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() == ".txt"
        and "raman_data_raw" in file.name.lower()
    )

    if not raw_files:
        raise RuntimeError("未找到 RAW 光谱文件")

    print("找到 RAW 光谱数量：", len(raw_files))

    # ======================================
    # 3. 创建数据容器
    # ======================================

    spectra = []

    materials = []
    sample_ids = []
    file_names = []

    # ======================================
    # 4. 批量预处理所有光谱
    # ======================================

    for i, file_path in enumerate(raw_files, start=1):

        print(
            f"处理第 {i}/{len(raw_files)} 条："
            f"{file_path.name}"
        )

        spectrum = load_spectrum(file_path)

        denoised = denoise_spectrum(
            spectrum,
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER
        )

        corrected = baseline_correct_spectrum(
            denoised,
            lam=ASLS_LAM,
            p=ASLS_P
        )

        normalized = normalize_spectrum(
            corrected,
            method="max"
        )

        # 保存预处理后的光谱对象
        spectra.append(normalized)

        # 提取标签与样品编号
        parts = file_path.name.split("__")

        materials.append(parts[0])
        sample_ids.append(parts[1])
        file_names.append(file_path.name)

    # ======================================
    # 5. 计算所有光谱的共同波数范围
    # ======================================

    common_min = max(
        spectrum.spectral_axis.min()
        for spectrum in spectra
    )

    common_max = min(
        spectrum.spectral_axis.max()
        for spectrum in spectra
    )

    print("\n共同波数下限：", common_min)
    print("共同波数上限：", common_max)

    # ======================================
    # 6. 创建共同拉曼位移网格
    # ======================================

    start = (
        np.ceil(common_min / GRID_STEP)
        * GRID_STEP
    )

    stop = (
        np.floor(common_max / GRID_STEP)
        * GRID_STEP
    )

    if stop <= start:
        raise ValueError(
            "共同波数范围不足，无法建立统一网格"
        )

    common_x = np.arange(
        start,
        stop + GRID_STEP / 2,
        GRID_STEP
    )

    print("共同网格点数：", len(common_x))

    # ======================================
    # 7. 将所有光谱插值到相同网格
    # ======================================

    aligned_spectra = []

    for spectrum in spectra:

        x = spectrum.spectral_axis
        y = spectrum.spectral_data

        aligned_y = np.interp(
            common_x,
            x,
            y
        )

        aligned_spectra.append(aligned_y)

    # ======================================
    # 8. 构建二维 NumPy 矩阵
    # ======================================

    X = np.vstack(aligned_spectra)

    print("\n===== 光谱矩阵 =====")
    print("矩阵形状：", X.shape)
    print("矩阵数据类型：", X.dtype)

    # 验证矩阵
    assert X.shape[0] == len(raw_files)
    assert X.shape[1] == len(common_x)
    assert np.all(np.isfinite(X))

    # ======================================
    # 9. 保存矩阵与元数据
    # ======================================

    np.savez_compressed(
        output_file,

        X=X,
        raman_shift=common_x,

        materials=np.array(materials),
        sample_ids=np.array(sample_ids),
        file_names=np.array(file_names),

        sg_window_length=SG_WINDOW,
        sg_polyorder=SG_POLYORDER,
        asls_lam=ASLS_LAM,
        asls_p=ASLS_P,
        grid_step=GRID_STEP
    )

    print("\n矩阵保存成功：", output_file)

    # ======================================
    # 10. 重新加载并验证保存结果
    # ======================================

    with np.load(
        output_file,
        allow_pickle=False
    ) as saved:

        saved_X = saved["X"]
        saved_axis = saved["raman_shift"]
        saved_materials = saved["materials"]

        np.testing.assert_array_equal(
            saved_X,
            X
        )

        np.testing.assert_array_equal(
            saved_axis,
            common_x
        )

        np.testing.assert_array_equal(
            saved_materials,
            np.array(materials)
        )

    print("文件重新加载验证通过！")
    print("材料类别：", sorted(set(materials)))
    print("\n实验 6.13 完成！")


if __name__ == "__main__":
    main()
