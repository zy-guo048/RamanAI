
"""RamanAI: reusable Raman spectroscopy processing functions."""

from pathlib import Path

import numpy as np
import ramanspy as rp
from scipy.signal import find_peaks, peak_widths


def load_spectrum(file_path):
    """Read a two-column, comma-separated RRUFF-style TXT spectrum."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到光谱文件：{file_path}")

    data = np.loadtxt(file_path, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("光谱文件应包含两列：拉曼位移、强度")
    if not np.all(np.isfinite(data)):
        raise ValueError("光谱文件包含无效数值")

    x = data[:, 0]
    y = data[:, 1]
    if not np.all(np.diff(x) > 0):
        raise ValueError("拉曼位移必须严格递增")

    return rp.Spectrum(y, x)


def denoise_spectrum(spectrum, window_length=9, polyorder=3):
    """Apply Savitzky-Golay smoothing to a RamanSPy Spectrum."""
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length 必须为正奇数")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder 必须非负且小于 window_length")
    if window_length > len(spectrum.spectral_axis):
        raise ValueError("window_length 不能超过光谱数据点数")

    denoiser = rp.preprocessing.denoise.SavGol(
        window_length=window_length,
        polyorder=polyorder,
    )
    return denoiser.apply(spectrum)


def baseline_correct_spectrum(spectrum, lam=1e6, p=0.01):
    """Apply asymmetric least squares (AsLS) baseline correction."""
    if lam <= 0:
        raise ValueError("lam 必须大于 0")
    if not 0 < p < 1:
        raise ValueError("p 必须在 0 与 1 之间")

    corrector = rp.preprocessing.baseline.ASLS(lam=lam, p=p)
    return corrector.apply(spectrum)


def detect_peaks(spectrum, prominence=1000, distance=10):
    """Return (peak indices, SciPy peak properties) for a spectrum."""
    if prominence < 0:
        raise ValueError("prominence 不能为负数")
    if distance < 1:
        raise ValueError("distance 必须至少为 1 个采样点")

    return find_peaks(
        spectrum.spectral_data,
        prominence=prominence,
        distance=distance,
    )


def extract_peak_features(spectrum, peaks, properties):
    """Extract positions, heights, prominences, and half-prominence widths.

    Widths are measured at half *prominence*, not fitted single-peak FWHM.
    The three extra fields are used to draw horizontal width markers.
    """
    x = spectrum.spectral_axis
    y = spectrum.spectral_data
    if not np.all(np.diff(x) > 0):
        raise ValueError("拉曼位移必须严格递增")
    if len(peaks) != len(properties["prominences"]):
        raise ValueError("峰索引与峰突出度数量不一致")

    widths = peak_widths(y, peaks, rel_height=0.5)
    indices = np.arange(len(x))
    left_cm = np.interp(widths[2], indices, x)
    right_cm = np.interp(widths[3], indices, x)

    features = []
    for i, peak in enumerate(peaks):
        features.append({
            "Peak_ID": i + 1,
            "Position_cm-1": float(x[peak]),
            "Intensity": float(y[peak]),
            "Prominence": float(properties["prominences"][i]),
            "Half_Prominence_Width_cm-1": float(right_cm[i] - left_cm[i]),
            "Left_cm-1": float(left_cm[i]),
            "Right_cm-1": float(right_cm[i]),
            "Width_Height": float(widths[1][i]),
        })
    return features


# ==========================================
# 6. 拉曼光谱归一化
# ==========================================

def normalize_spectrum(spectrum, method="max"):
    """
    对拉曼光谱进行强度归一化。

    参数:
        spectrum: RamanSPy Spectrum 对象
        method: 归一化方法，目前支持 "max"

    返回:
        normalized: 归一化后的 Spectrum 对象
    """

    # 1. 检查归一化方法
    if method != "max":
        raise ValueError(
            f"不支持的归一化方法: {method}"
        )

    # 2. 获取光谱数据
    x = spectrum.spectral_axis
    y = spectrum.spectral_data

    # 3. 检查强度数据
    if not np.all(np.isfinite(y)):
        raise ValueError(
            "光谱包含无效强度数据"
        )

    # 4. 找到最大强度
    max_intensity = np.max(y)

    if max_intensity <= 0:
        raise ValueError(
            "光谱最大强度必须大于 0"
        )

    # 5. 最大值归一化
    normalized_y = y / max_intensity

    # 6. 创建新的 RamanSPy 光谱对象
    normalized = rp.Spectrum(
        normalized_y,
        x
    )

    return normalized
