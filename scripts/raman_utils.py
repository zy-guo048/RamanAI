from pathlib import Path
import numpy as np
import ramanspy as rp


def load_spectrum(file_path):
    """
    读取两列逗号分隔的 RRUFF 拉曼光谱文件。

    参数:
        file_path: 光谱文件路径

    返回:
        RamanSPy Spectrum 对象
    """

    # 1. 将文件路径转换为 Path 对象
    file_path = Path(file_path)

    # 2. 检查文件是否存在
    if not file_path.is_file():
        raise FileNotFoundError(
            f"找不到光谱文件: {file_path}"
        )

    # 3. 读取数据
    data = np.loadtxt(
        file_path,
        delimiter=",",
        comments="#"
    )

    # 4. 检查数据格式
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            "光谱数据必须包含两列：波数和强度"
        )

    # 5. 提取波数和强度
    raman_shift = data[:, 0]
    intensity = data[:, 1]

    # 6. 创建光谱对象
    spectrum = rp.Spectrum(
        intensity,
        raman_shift
    )

    # 7. 返回结果
    return spectrum