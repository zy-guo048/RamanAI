"""Single-spectrum Raman analysis pipeline for uploaded files."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import ramanspy as rp

from raman_app.upload import parse_uploaded_spectrum
from scripts.raman_utils import (
    baseline_correct_spectrum,
    denoise_spectrum,
    detect_peaks,
    extract_peak_features,
    normalize_spectrum,
)


DEFAULT_PEAK_PROMINENCE = 1000.0


@dataclass(frozen=True)
class AnalysisParameters:
    """Parameters used by the stable RamanAI V1 analysis workflow."""

    sg_window_length: int = 21
    sg_polyorder: int = 3
    asls_lam: float = 1e6
    asls_p: float = 0.01
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE
    peak_distance: int = 10
    peak_distance_unit: str = "samples"
    peak_width_method: str = "half_prominence"
    peak_width_rel_height: float = 0.5
    normalization_method: str = "max"


@dataclass
class AnalysisResult:
    """Structured output from a single uploaded-spectrum analysis."""

    filename: str
    point_count: int
    raw_spectrum: rp.Spectrum
    denoised_spectrum: rp.Spectrum
    corrected_spectrum: rp.Spectrum
    normalized_spectrum: rp.Spectrum
    peak_indices: np.ndarray
    peak_count: int
    peak_features: list[dict[str, int | float]]
    parameters: AnalysisParameters


def analyze_uploaded_spectrum(
    file_bytes: bytes,
    filename: str = "sample_raw.txt",
    prominence: float = DEFAULT_PEAK_PROMINENCE,
) -> AnalysisResult:
    """Run the stable RamanAI analysis workflow on an uploaded TXT file."""
    _validate_filename(filename)
    validated_prominence = _validate_prominence(prominence)
    parameters = AnalysisParameters(peak_prominence=validated_prominence)

    raw_spectrum = parse_uploaded_spectrum(file_bytes)
    denoised_spectrum = denoise_spectrum(
        raw_spectrum,
        window_length=parameters.sg_window_length,
        polyorder=parameters.sg_polyorder,
    )
    corrected_spectrum = baseline_correct_spectrum(
        denoised_spectrum,
        lam=parameters.asls_lam,
        p=parameters.asls_p,
    )
    peak_indices, peak_properties = detect_peaks(
        corrected_spectrum,
        prominence=parameters.peak_prominence,
        distance=parameters.peak_distance,
    )
    peak_features = extract_peak_features(
        corrected_spectrum,
        peak_indices,
        peak_properties,
    )
    normalized_spectrum = normalize_spectrum(
        corrected_spectrum,
        method=parameters.normalization_method,
    )

    return AnalysisResult(
        filename=filename,
        point_count=len(raw_spectrum.spectral_axis),
        raw_spectrum=raw_spectrum,
        denoised_spectrum=denoised_spectrum,
        corrected_spectrum=corrected_spectrum,
        normalized_spectrum=normalized_spectrum,
        peak_indices=peak_indices,
        peak_count=len(peak_indices),
        peak_features=peak_features,
        parameters=parameters,
    )


def _validate_filename(filename: str) -> None:
    if not isinstance(filename, str):
        raise TypeError("文件名必须是字符串")
    if not filename or not filename.strip():
        raise ValueError("文件名不能为空")
    if filename != filename.strip():
        raise ValueError("文件名首尾不能包含空白字符")
    if "/" in filename or "\\" in filename or "\x00" in filename:
        raise ValueError("文件名不能包含路径或无效字符")
    if filename.casefold() == ".txt" or not filename.casefold().endswith(".txt"):
        raise ValueError("仅支持扩展名为 .txt 的光谱文件")


def _validate_prominence(prominence: float) -> float:
    if isinstance(prominence, bool) or not isinstance(prominence, Real):
        raise TypeError("prominence 必须是数值")

    value = float(prominence)
    if not np.isfinite(value):
        raise ValueError("prominence 必须是有限数值")
    if value < 0:
        raise ValueError("prominence 不能为负数")
    return value
