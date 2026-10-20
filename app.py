"""Streamlit interface for RamanAI V1 single-spectrum analysis."""

import hashlib
import logging

import matplotlib.pyplot as plt

# 配置 Matplotlib 中文字体
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans"
]

# 避免负号显示异常
plt.rcParams["axes.unicode_minus"] = False


import pandas as pd
import streamlit as st

from raman_app.export import build_peak_csv_filename, peak_features_to_csv
from raman_app.pipeline import (
    DEFAULT_PEAK_PROMINENCE,
    AnalysisResult,
    analyze_uploaded_spectrum,
)


LOGGER = logging.getLogger(__name__)
ANALYSIS_RESULT_STATE_KEY = "ramanai_analysis_result"
ANALYSIS_REQUEST_STATE_KEY = "ramanai_analysis_request"


PEAK_TABLE_COLUMNS = [
    "Peak_ID",
    "Position_cm-1",
    "Intensity",
    "Prominence",
    "Half_Prominence_Width_cm-1",
]
PEAK_TABLE_LABELS = {
    "Peak_ID": "峰编号",
    "Position_cm-1": "峰位 (cm⁻¹)",
    "Intensity": "校正后强度 (a.u.)",
    "Prominence": "突出度",
    "Half_Prominence_Width_cm-1": "半突出度宽度 (cm⁻¹)",
}


def main() -> None:
    st.set_page_config(
        page_title="RamanAI",
        page_icon="🔬",
        layout="wide",
    )
    st.title("RamanAI — 智能拉曼光谱分析系统")
    st.caption("上传单条 RRUFF RAW TXT 光谱，使用固定预处理参数和可调突出度进行分析。")
    _initialize_analysis_state()

    uploaded_file = st.file_uploader(
        "上传 RRUFF RAW 光谱",
        type=["txt"],
        key="spectrum_upload",
        help="文件应包含 # 注释行和逗号分隔的两列数值。",
    )
    prominence = st.number_input(
        "候选峰突出度阈值",
        min_value=0.0,
        value=DEFAULT_PEAK_PROMINENCE,
        step=10.0,
        format="%.1f",
        key="peak_prominence",
        help="该阈值应用于未归一化的 AsLS 基线校正光谱。",
        icon=":material/tune:",
    )
    st.warning(
        "较低的突出度阈值可能引入噪声峰；检测结果仅表示候选峰。"
    )
    submitted = st.button(
        "开始分析",
        type="primary",
        key="analyze_spectrum",
        icon=":material/analytics:",
    )

    if uploaded_file is None:
        _clear_analysis_state()
        st.info("请选择一个 .txt 光谱文件，然后点击“开始分析”。")
        return

    file_bytes = uploaded_file.getvalue()
    request_key = _build_analysis_request_key(
        file_bytes,
        uploaded_file.name,
        prominence,
    )
    if st.session_state[ANALYSIS_REQUEST_STATE_KEY] != request_key:
        _clear_analysis_state()

    st.write(f"已选择文件：`{uploaded_file.name}`")
    if submitted:
        try:
            with st.spinner("正在分析光谱……"):
                result = analyze_uploaded_spectrum(
                    file_bytes,
                    filename=uploaded_file.name,
                    prominence=prominence,
                )
        except (TypeError, ValueError) as exc:
            _clear_analysis_state()
            st.error(str(exc), title="无法分析上传文件")
        except Exception:
            _clear_analysis_state()
            LOGGER.exception(
                "Unexpected Raman analysis error for uploaded file %s",
                uploaded_file.name,
            )
            st.error(
                "光谱处理过程中发生内部错误，详细信息已记录到服务器日志。",
                title="内部错误",
            )
        else:
            st.session_state[ANALYSIS_RESULT_STATE_KEY] = result
            st.session_state[ANALYSIS_REQUEST_STATE_KEY] = request_key

    result = st.session_state[ANALYSIS_RESULT_STATE_KEY]
    if result is not None:
        _render_result(result)


def _render_result(result: AnalysisResult) -> None:
    st.success("光谱分析完成。")

    file_column, points_column, peaks_column = st.columns(3)
    file_column.metric("文件名", result.filename)
    points_column.metric("数据点数", result.point_count)
    peaks_column.metric("候选峰数量", result.peak_count)

    _render_parameters(result)

    st.subheader("预处理光谱")
    processing_figure = _create_processing_figure(result)
    st.pyplot(
        processing_figure,
        clear_figure=True,
        alt="原始、SG 去噪和 AsLS 基线校正光谱",
    )
    plt.close(processing_figure)

    st.subheader("最大值归一化光谱")
    normalized_figure = _create_normalized_figure(result)
    st.pyplot(
        normalized_figure,
        clear_figure=True,
        alt="最大值归一化后的基线校正光谱",
    )
    plt.close(normalized_figure)

    st.subheader("基线校正光谱与候选峰")
    peak_figure = _create_peak_figure(result)
    st.pyplot(
        peak_figure,
        clear_figure=True,
        alt="基线校正光谱及其候选峰标记",
    )
    plt.close(peak_figure)

    st.subheader("候选峰特征")
    peak_table = _build_peak_table(result)
    if result.peak_count == 0:
        st.info("当前参数下未检测到候选峰。")
    st.dataframe(
        peak_table,
        width="stretch",
        hide_index=True,
        alt="候选峰的峰位、校正后强度、突出度和半突出度宽度",
    )
    st.caption("候选峰为算法检测结果，不代表矿物分类或分类准确率结论。")
    st.download_button(
        "下载候选峰特征 CSV",
        data=peak_features_to_csv(result.peak_features),
        file_name=build_peak_csv_filename(result.filename),
        mime="text/csv;charset=utf-8",
        key="download_peak_features",
        on_click="ignore",
        icon=":material/download:",
    )


def _render_parameters(result: AnalysisResult) -> None:
    parameters = result.parameters
    with st.expander("本次分析参数（未自动优化）", expanded=True):
        st.markdown(
            "\n".join([
                f"- SG 去噪：window_length={parameters.sg_window_length}，"
                f"polyorder={parameters.sg_polyorder}",
                f"- AsLS 基线校正：lam={parameters.asls_lam:g}，"
                f"p={parameters.asls_p:g}",
                f"- 候选峰检测：prominence={parameters.peak_prominence:g}，"
                f"distance={parameters.peak_distance} 个采样点",
                "- 峰宽：半突出度宽度，rel_height=0.5",
                f"- 归一化：{parameters.normalization_method}",
            ])
        )
        st.caption("寻峰在未归一化的 AsLS 基线校正光谱上执行。")


def _create_processing_figure(result: AnalysisResult):
    spectra = [
        (result.raw_spectrum, "原始光谱", "Raw Intensity (a.u.)", "#4C78A8"),
        (
            result.denoised_spectrum,
            "SG 去噪光谱",
            "SG-Denoised Intensity (a.u.)",
            "#F58518",
        ),
        (
            result.corrected_spectrum,
            "AsLS 基线校正光谱",
            "Baseline-Corrected Intensity (a.u.)",
            "#54A24B",
        ),
    ]
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    for axis, (spectrum, title, ylabel, color) in zip(axes, spectra):
        axis.plot(
            spectrum.spectral_axis,
            spectrum.spectral_data,
            color=color,
            linewidth=1.1,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Raman Shift (cm$^{-1}$)")
    return figure


def _create_normalized_figure(result: AnalysisResult):
    spectrum = result.normalized_spectrum
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(
        spectrum.spectral_axis,
        spectrum.spectral_data,
        color="#B279A2",
        linewidth=1.1,
    )
    axis.set_title("最大值归一化后的基线校正光谱")
    axis.set_xlabel("Raman Shift (cm$^{-1}$)")
    axis.set_ylabel("Max-Normalized Intensity")
    axis.grid(alpha=0.2)
    return figure


def _create_peak_figure(result: AnalysisResult):
    spectrum = result.corrected_spectrum
    raman_shift = spectrum.spectral_axis
    intensity = spectrum.spectral_data

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(
        raman_shift,
        intensity,
        color="#4C78A8",
        linewidth=1.1,
        label="Baseline-Corrected Spectrum",
    )
    if result.peak_count:
        axis.scatter(
            raman_shift[result.peak_indices],
            intensity[result.peak_indices],
            color="#E45756",
            s=38,
            zorder=3,
            label="Detected Candidate Peaks",
        )
    axis.set_title("AsLS 基线校正光谱上的候选峰")
    axis.set_xlabel("Raman Shift (cm$^{-1}$)")
    axis.set_ylabel("Baseline-Corrected Intensity (a.u.)")
    axis.grid(alpha=0.2)
    axis.legend()
    return figure


def _build_peak_table(result: AnalysisResult) -> pd.DataFrame:
    table = pd.DataFrame.from_records(
        result.peak_features,
        columns=PEAK_TABLE_COLUMNS,
    )
    return table.rename(columns=PEAK_TABLE_LABELS)


def _initialize_analysis_state() -> None:
    st.session_state.setdefault(ANALYSIS_RESULT_STATE_KEY, None)
    st.session_state.setdefault(ANALYSIS_REQUEST_STATE_KEY, None)


def _clear_analysis_state() -> None:
    st.session_state[ANALYSIS_RESULT_STATE_KEY] = None
    st.session_state[ANALYSIS_REQUEST_STATE_KEY] = None


def _build_analysis_request_key(
    file_bytes: bytes,
    filename: str,
    prominence: object,
) -> tuple[str, str, str]:
    digest = hashlib.sha256(file_bytes).hexdigest()
    prominence_identity = f"{type(prominence).__name__}:{prominence!r}"
    return filename, digest, prominence_identity


if __name__ == "__main__":
    main()
