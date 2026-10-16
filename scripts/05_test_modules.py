
"""Experiment 5: verify that RamanAI wrappers match direct library calls.

Historical peak positions/widths are intentionally NOT asserted here.
This tests correct encapsulation on the CURRENT spectrum and parameters.
"""

from pathlib import Path

import numpy as np
import ramanspy as rp
from scipy.signal import find_peaks, peak_widths

from raman_utils import (
    load_spectrum,
    denoise_spectrum,
    baseline_correct_spectrum,
    detect_peaks,
    extract_peak_features,
)


def main():
    project_dir = Path(__file__).resolve().parent.parent
    file_path = project_dir / "data" / "raw" / "sample_raw.txt"

    # 1. Check the custom reader against the raw data in the current file.
    spectrum = load_spectrum(file_path)
    raw = np.loadtxt(file_path, delimiter=",", comments="#")
    np.testing.assert_array_equal(spectrum.spectral_axis, raw[:, 0])
    np.testing.assert_array_equal(spectrum.spectral_data, raw[:, 1])
    print(f"[通过] 光谱读取：{len(spectrum.spectral_axis)} 个数据点")

    # 2. Compare the denoising wrapper with the direct RamanSPy call.
    direct_denoised = rp.preprocessing.denoise.SavGol(
        window_length=21, polyorder=3
    ).apply(spectrum)
    wrapped_denoised = denoise_spectrum(
        spectrum, window_length=21, polyorder=3
    )
    np.testing.assert_allclose(
        wrapped_denoised.spectral_data,
        direct_denoised.spectral_data,
        rtol=0, atol=1e-8,
    )
    print("[通过] SG 去噪：封装结果与直接调用一致")

    # 3. Compare the baseline-correction wrapper with a direct call.
    direct_corrected = rp.preprocessing.baseline.ASLS(
        lam=1e6, p=0.01
    ).apply(wrapped_denoised)
    corrected = baseline_correct_spectrum(
        wrapped_denoised, lam=1e6, p=0.01
    )
    np.testing.assert_allclose(
        corrected.spectral_data,
        direct_corrected.spectral_data,
        rtol=0, atol=1e-8,
    )
    print("[通过] AsLS 基线校正：封装结果与直接调用一致")

    # 4. Compare peak indices and properties with SciPy directly.
    y = corrected.spectral_data
    x = corrected.spectral_axis
    direct_peaks, direct_props = find_peaks(
        y, prominence=1000, distance=10
    )
    peaks, props = detect_peaks(
        corrected, prominence=1000, distance=10
    )
    np.testing.assert_array_equal(peaks, direct_peaks)
    np.testing.assert_allclose(
        props["prominences"], direct_props["prominences"],
        rtol=0, atol=1e-8,
    )
    print("[通过] 自动寻峰：峰索引和突出度一致")

    # 5. Compare extracted features with a direct SciPy calculation.
    features = extract_peak_features(corrected, peaks, props)
    widths = peak_widths(y, peaks, rel_height=0.5)
    indices = np.arange(len(x))
    left_cm = np.interp(widths[2], indices, x)
    right_cm = np.interp(widths[3], indices, x)
    expected_width_cm = right_cm - left_cm

    feature_positions = np.array([f["Position_cm-1"] for f in features])
    feature_intensities = np.array([f["Intensity"] for f in features])
    feature_prominences = np.array([f["Prominence"] for f in features])
    feature_widths = np.array([
        f["Half_Prominence_Width_cm-1"] for f in features
    ])

    assert len(features) == len(peaks)
    np.testing.assert_allclose(feature_positions, x[peaks], rtol=0, atol=1e-8)
    np.testing.assert_allclose(feature_intensities, y[peaks], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        feature_prominences, props["prominences"], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        feature_widths, expected_width_cm, rtol=0, atol=1e-8
    )
    print("[通过] 特征提取：峰位、强度、突出度和半突出度宽度一致")

    print("\n===== 当前光谱结果（不是历史数值的断言）=====")
    print("峰数量：", len(features))
    print("峰位：", np.round(feature_positions, 3))
    print("半突出度宽度：", np.round(feature_widths, 3))
    print("\n全部模块测试通过！")


if __name__ == "__main__":
    main()



