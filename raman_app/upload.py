"""Parse and validate uploaded RRUFF Raman spectra in memory."""

from io import StringIO

import numpy as np
import ramanspy as rp


MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
MIN_SPECTRUM_POINTS = 21


def parse_uploaded_spectrum(file_bytes: bytes) -> rp.Spectrum:
    """Parse an uploaded RRUFF-style TXT spectrum from binary content."""
    if not isinstance(file_bytes, bytes):
        raise TypeError("上传内容必须是 bytes 类型")
    if not file_bytes or not file_bytes.strip():
        raise ValueError("上传文件为空")
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError("上传文件超过 5 MB 大小限制")

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("上传文件必须是 UTF-8 编码的文本文件") from exc

    data_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        raise ValueError("上传文件不包含光谱数据")

    try:
        data = np.loadtxt(StringIO(text), delimiter=",", comments="#")
    except (TypeError, ValueError) as exc:
        raise ValueError("光谱数据格式错误：应为逗号分隔的两列数值") from exc

    if data.ndim == 1 and data.size == 2:
        data = data.reshape(1, 2)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("光谱数据必须包含两列：拉曼位移、强度")
    if not np.all(np.isfinite(data)):
        raise ValueError("光谱数据包含 NaN 或 Inf 等无效数值")
    if data.shape[0] < MIN_SPECTRUM_POINTS:
        raise ValueError(
            f"光谱数据点不足：至少需要 {MIN_SPECTRUM_POINTS} 个采样点"
        )

    raman_shift = data[:, 0]
    intensity = data[:, 1]
    if not np.all(np.diff(raman_shift) > 0):
        raise ValueError("拉曼位移必须严格递增，不能重复或降序")

    return rp.Spectrum(intensity, raman_shift)
