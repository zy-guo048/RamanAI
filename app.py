"""Streamlit interface for RamanAI V1 single-spectrum analysis."""

import hashlib
import logging
from pathlib import Path

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
from raman_app.i18n import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    language_label,
    normalize_language,
    translate,
    translate_error,
    translate_matching_error,
)
from raman_app.matching import (
    ALLOWED_TOP_K,
    DEFAULT_CACHE_PATH,
    DEFAULT_INDEX_PATH,
    SpectrumMatchingResult,
    build_matching_csv_filename,
    build_matching_json_filename,
    fingerprint_reference_sources,
    load_reference_library_with_fingerprint,
    match_uploaded_spectrum,
    matching_exports_for_current_source,
    matching_result_matches_request,
)
from raman_app.pipeline import (
    DEFAULT_PEAK_PROMINENCE,
    AnalysisResult,
    analyze_uploaded_spectrum,
)


LOGGER = logging.getLogger(__name__)
ANALYSIS_RESULT_STATE_KEY = "ramanai_analysis_result"
ANALYSIS_REQUEST_STATE_KEY = "ramanai_analysis_request"
LANGUAGE_STATE_KEY = "ramanai_language"
LANGUAGE_SELECTOR_STATE_KEY = "ramanai_language_selector"
MAIN_VIEW_STATE_KEY = "ramanai_main_view"
MATCHING_RESULT_STATE_KEY = "ramanai_matching_result"
MATCHING_REQUEST_STATE_KEY = "ramanai_matching_request"


PEAK_TABLE_COLUMNS = [
    "Peak_ID",
    "Position_cm-1",
    "Intensity",
    "Prominence",
    "Half_Prominence_Width_cm-1",
]
PEAK_TABLE_LABEL_KEYS = {
    "Peak_ID": "table_peak_id",
    "Position_cm-1": "table_position",
    "Intensity": "table_intensity",
    "Prominence": "table_prominence",
    "Half_Prominence_Width_cm-1": "table_width",
}


@st.dialog("⚙️ 设置 / Settings", icon=":material/settings:")
def _show_settings_dialog() -> None:
    language = _current_language()
    if LANGUAGE_SELECTOR_STATE_KEY not in st.session_state:
        st.session_state[LANGUAGE_SELECTOR_STATE_KEY] = language
    st.radio(
        translate(language, "language"),
        options=SUPPORTED_LANGUAGES,
        format_func=language_label,
        key=LANGUAGE_SELECTOR_STATE_KEY,
        on_change=_apply_language_selection,
    )
    st.caption(translate(language, "language_note"))


def main() -> None:
    st.set_page_config(
        page_title="RamanAI",
        page_icon="🔬",
        layout="wide",
    )
    _initialize_analysis_state()
    _initialize_matching_state()
    language = _current_language()

    title_column, settings_column = st.columns(
        [5, 1],
        vertical_alignment="center",
    )
    with title_column:
        st.title(translate(language, "app_title"))
    with settings_column:
        if st.button(
            translate(language, "settings"),
            key="open_settings",
            width="stretch",
        ):
            _show_settings_dialog()
    analysis_tab, matching_tab = st.tabs(
        [
            translate(language, "analysis_view"),
            translate(language, "matching_view"),
        ],
        key=MAIN_VIEW_STATE_KEY,
        on_change="rerun",
    )
    if analysis_tab.open:
        with analysis_tab:
            st.caption(translate(language, "app_caption"))
            _render_analysis_page(language)
    if matching_tab.open:
        with matching_tab:
            st.caption(translate(language, "matching_caption"))
            _render_matching_page(language)


def _render_analysis_page(language: str) -> None:

    uploaded_file = st.file_uploader(
        translate(language, "upload_file"),
        type=["txt"],
        key="spectrum_upload",
        help=translate(language, "upload_help"),
    )
    prominence = st.number_input(
        translate(language, "peak_prominence"),
        min_value=0.0,
        value=DEFAULT_PEAK_PROMINENCE,
        step=10.0,
        format="%.1f",
        key="peak_prominence",
        help=translate(language, "peak_prominence_help"),
        icon=":material/tune:",
    )
    st.warning(translate(language, "prominence_warning"))
    submitted = st.button(
        translate(language, "start_analysis"),
        type="primary",
        key="analyze_spectrum",
        icon=":material/analytics:",
    )

    if uploaded_file is None:
        _clear_analysis_state()
        st.info(translate(language, "select_file_prompt"))
        return

    file_bytes = uploaded_file.getvalue()
    request_key = _build_analysis_request_key(
        file_bytes,
        uploaded_file.name,
        prominence,
    )
    if st.session_state[ANALYSIS_REQUEST_STATE_KEY] != request_key:
        _clear_analysis_state()

    st.markdown(
        translate(language, "selected_file", filename=uploaded_file.name)
    )
    if submitted:
        try:
            with st.spinner(translate(language, "analyzing")):
                result = analyze_uploaded_spectrum(
                    file_bytes,
                    filename=uploaded_file.name,
                    prominence=prominence,
                )
        except (TypeError, ValueError) as exc:
            _clear_analysis_state()
            st.error(
                translate_error(language, str(exc)),
                title=translate(language, "input_error_title"),
            )
        except Exception:
            _clear_analysis_state()
            LOGGER.exception(
                "Unexpected Raman analysis error for uploaded file %s",
                uploaded_file.name,
            )
            st.error(
                translate(language, "internal_error_message"),
                title=translate(language, "internal_error_title"),
            )
        else:
            st.session_state[ANALYSIS_RESULT_STATE_KEY] = result
            st.session_state[ANALYSIS_REQUEST_STATE_KEY] = request_key

    result = st.session_state[ANALYSIS_RESULT_STATE_KEY]
    if result is not None:
        _render_result(result, language)


@st.cache_resource(max_entries=2, show_spinner=False)
def _load_matching_library_cached(
    cache_path: str,
    index_path: str,
    cache_size: int,
    cache_mtime_ns: int,
    index_size: int,
    index_mtime_ns: int,
    cache_sha256: str,
    index_sha256: str,
):
    del (
        cache_size,
        cache_mtime_ns,
        index_size,
        index_mtime_ns,
        cache_sha256,
        index_sha256,
    )
    return load_reference_library_with_fingerprint(
        Path(cache_path), Path(index_path)
    )


def _get_matching_library():
    if not DEFAULT_CACHE_PATH.is_file():
        _load_matching_library_cached.clear()
        raise ValueError(f"NPZ cache does not exist: {DEFAULT_CACHE_PATH}")
    if not DEFAULT_INDEX_PATH.is_file():
        _load_matching_library_cached.clear()
        raise ValueError(f"Index CSV does not exist: {DEFAULT_INDEX_PATH}")
    current_source = fingerprint_reference_sources(
        DEFAULT_CACHE_PATH,
        DEFAULT_INDEX_PATH,
    )
    try:
        bundle = _load_matching_library_cached(
            str(DEFAULT_CACHE_PATH),
            str(DEFAULT_INDEX_PATH),
            current_source.npz_cache.size_bytes,
            current_source.npz_cache.mtime_ns,
            current_source.index_csv.size_bytes,
            current_source.index_csv.mtime_ns,
            current_source.npz_cache.sha256,
            current_source.index_csv.sha256,
        )
    except Exception:
        _load_matching_library_cached.clear()
        raise
    if bundle.source_fingerprint != current_source:
        _load_matching_library_cached.clear()
        raise ValueError(
            "Cached reference library does not match current source content"
        )
    return bundle


def _render_matching_page(language: str) -> None:
    uploaded_file = st.file_uploader(
        translate(language, "matching_upload"),
        type=["txt"],
        key="matching_spectrum_upload",
        help=translate(language, "matching_upload_help"),
    )
    top_k = st.segmented_control(
        translate(language, "matching_top_k"),
        options=ALLOWED_TOP_K,
        default=5,
        key="matching_top_k",
        required=True,
    )
    submitted = st.button(
        translate(language, "start_matching"),
        type="primary",
        key="match_spectrum",
        icon=":material/search:",
    )
    st.info(translate(language, "matching_application_library_note"))
    if uploaded_file is None:
        _clear_matching_state()
        st.info(translate(language, "matching_select_prompt"))
        return

    file_bytes = uploaded_file.getvalue()
    request_key = _build_matching_request_key(
        file_bytes,
        uploaded_file.name,
        int(top_k),
    )
    if st.session_state[MATCHING_REQUEST_STATE_KEY] != request_key:
        _clear_matching_state()
    st.markdown(
        translate(language, "matching_selected_file", filename=uploaded_file.name)
    )
    if submitted:
        try:
            with st.spinner(translate(language, "matching_running")):
                library_bundle = _get_matching_library()
                result = match_uploaded_spectrum(
                    file_bytes,
                    uploaded_file.name,
                    int(top_k),
                    library_bundle.library,
                    source_fingerprint=library_bundle.source_fingerprint,
                )
        except (TypeError, ValueError, OSError) as exc:
            _clear_matching_state()
            st.error(
                translate_matching_error(language, str(exc)),
                title=translate(language, "matching_input_error_title"),
            )
        except Exception:
            _clear_matching_state()
            LOGGER.exception(
                "Unexpected Raman matching error for uploaded file %s",
                uploaded_file.name,
            )
            st.error(
                translate(language, "matching_internal_error_message"),
                title=translate(language, "matching_internal_error_title"),
            )
        else:
            st.session_state[MATCHING_RESULT_STATE_KEY] = result
            st.session_state[MATCHING_REQUEST_STATE_KEY] = request_key

    result = st.session_state[MATCHING_RESULT_STATE_KEY]
    if result is not None:
        if matching_result_matches_request(
            result,
            file_bytes,
            uploaded_file.name,
            int(top_k),
        ):
            try:
                csv_content, json_content = matching_exports_for_current_source(
                    result,
                    DEFAULT_CACHE_PATH,
                    DEFAULT_INDEX_PATH,
                )
            except (TypeError, ValueError, OSError) as exc:
                _load_matching_library_cached.clear()
                _clear_matching_state()
                st.error(
                    translate_matching_error(language, str(exc)),
                    title=translate(language, "matching_input_error_title"),
                )
            else:
                _render_matching_result(
                    result,
                    language,
                    csv_content,
                    json_content,
                )
        else:
            _clear_matching_state()


def _render_matching_result(
    result: SpectrumMatchingResult,
    language: str,
    csv_content: bytes,
    json_content: bytes,
) -> None:
    st.success(translate(language, "matching_complete"))
    file_column, library_column, candidate_column = st.columns(3)
    file_column.metric(translate(language, "filename"), result.filename)
    library_column.metric(
        translate(language, "matching_reference_library"),
        result.reference_library_size,
    )
    candidate_column.metric(
        translate(language, "matching_top_k"), len(result.candidates)
    )

    with st.expander(translate(language, "matching_parameters"), expanded=True):
        st.markdown("- " + translate(language, "matching_parameter_flow"))
        st.markdown("- " + translate(language, "matching_grid"))

    st.subheader(translate(language, "matching_plot_section"))
    figure = _create_matching_figure(result, language)
    st.pyplot(
        figure,
        clear_figure=True,
        alt=translate(language, "matching_plot_alt"),
    )
    plt.close(figure)

    st.subheader(translate(language, "matching_candidates"))
    candidate_table = pd.DataFrame(
        [
            {
                translate(language, "match_rank"): candidate.rank,
                translate(language, "reference_label"): candidate.reference_label,
                translate(language, "cosine_similarity"): candidate.cosine_similarity,
                translate(language, "reference_file_count"): candidate.reference_file_count,
                translate(language, "distinct_identity_count"): candidate.distinguishable_identity_count,
                translate(language, "best_reference_file"): candidate.best_relative_path,
                translate(language, "best_sample_id"): candidate.best_sample_id,
                translate(language, "excitation_wavelength"): candidate.best_excitation_wavelength_nm,
                translate(language, "identity_score_gap"): candidate.best_minus_second_identity_score,
            }
            for candidate in result.candidates
        ]
    )
    st.dataframe(
        candidate_table,
        hide_index=True,
        alt=translate(language, "matching_candidate_table_alt"),
    )

    st.subheader(translate(language, "matching_downloads"))
    with st.container(horizontal=True):
        st.download_button(
            translate(language, "download_matching_csv"),
            data=csv_content,
            file_name=build_matching_csv_filename(result.filename),
            mime="text/csv;charset=utf-8",
            key="download_matching_candidates_csv",
            on_click="ignore",
            icon=":material/download:",
        )
        st.download_button(
            translate(language, "download_matching_json"),
            data=json_content,
            file_name=build_matching_json_filename(result.filename),
            mime="application/json;charset=utf-8",
            key="download_matching_record_json",
            on_click="ignore",
            icon=":material/download:",
        )
    st.caption(translate(language, "matching_download_note"))

    for candidate in result.candidates:
        expander = st.expander(
            f"#{candidate.rank} {candidate.reference_label} — "
            f"{candidate.cosine_similarity:.6f}"
        )
        with expander:
            st.markdown(
                f"**{translate(language, 'best_reference_file')}：** "
                f"`{candidate.best_relative_path}`"
            )
            support_table = pd.DataFrame(
                [
                    {
                        translate(language, "support_rank"): item.rank,
                        "Sample_ID": item.sample_id,
                        translate(language, "support_score"): item.score,
                        translate(language, "best_reference_file"): item.relative_path,
                        translate(language, "excitation_wavelength"): item.excitation_wavelength_nm,
                        translate(language, "support_file_count"): item.file_count,
                        translate(language, "identity_basis"): item.identity_basis,
                    }
                    for item in candidate.supporting_identities
                ]
            )
            st.dataframe(
                support_table,
                hide_index=True,
                alt=translate(language, "support_evidence"),
            )
    st.warning(translate(language, "matching_scientific_note"))


def _create_matching_figure(
    result: SpectrumMatchingResult,
    language: str,
):
    top_candidate = result.candidates[0]
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(result.grid, result.query_vector, color="#4C78A8", linewidth=1.1)
    axes[0].set_title(translate(language, "query_processed_title"))
    axes[1].plot(
        result.grid,
        top_candidate.best_reference_vector,
        color="#F58518",
        linewidth=1.1,
    )
    axes[1].set_title(
        translate(language, "reference_processed_title")
        + f" — {top_candidate.reference_label}"
    )
    axes[2].plot(
        result.grid,
        result.query_vector,
        color="#4C78A8",
        linewidth=1.1,
        label=translate(language, "query_spectrum_label"),
    )
    axes[2].plot(
        result.grid,
        top_candidate.best_reference_vector,
        color="#F58518",
        linewidth=1.1,
        alpha=0.85,
        label=translate(language, "reference_spectrum_label"),
    )
    axes[2].set_title(translate(language, "overlay_title"))
    axes[2].legend()
    for axis in axes:
        axis.set_ylabel(translate(language, "matching_intensity_axis"))
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel(translate(language, "raman_shift_axis"))
    return figure


def _render_result(result: AnalysisResult, language: str) -> None:
    st.success(translate(language, "analysis_complete"))

    file_column, points_column, peaks_column = st.columns(3)
    file_column.metric(translate(language, "filename"), result.filename)
    points_column.metric(translate(language, "point_count"), result.point_count)
    peaks_column.metric(translate(language, "peak_count"), result.peak_count)

    _render_parameters(result, language)

    st.subheader(translate(language, "preprocessed_spectra"))
    processing_figure = _create_processing_figure(result, language)
    st.pyplot(
        processing_figure,
        clear_figure=True,
        alt=translate(language, "processing_plot_alt"),
    )
    plt.close(processing_figure)

    st.subheader(translate(language, "normalized_spectrum_section"))
    normalized_figure = _create_normalized_figure(result, language)
    st.pyplot(
        normalized_figure,
        clear_figure=True,
        alt=translate(language, "normalized_plot_alt"),
    )
    plt.close(normalized_figure)

    st.subheader(translate(language, "peak_plot_section"))
    peak_figure = _create_peak_figure(result, language)
    st.pyplot(
        peak_figure,
        clear_figure=True,
        alt=translate(language, "peak_plot_alt"),
    )
    plt.close(peak_figure)

    st.subheader(translate(language, "peak_features"))
    peak_table = _build_peak_table(result, language)
    if result.peak_count == 0:
        st.info(translate(language, "no_peaks_found"))
    st.dataframe(
        peak_table,
        width="stretch",
        hide_index=True,
        alt=translate(language, "peak_table_alt"),
    )
    st.caption(translate(language, "candidate_disclaimer"))
    st.download_button(
        translate(language, "download_csv"),
        data=peak_features_to_csv(result.peak_features),
        file_name=build_peak_csv_filename(result.filename),
        mime="text/csv;charset=utf-8",
        key="download_peak_features",
        on_click="ignore",
        icon=":material/download:",
    )


def _render_parameters(result: AnalysisResult, language: str) -> None:
    parameters = result.parameters
    with st.expander(
        translate(language, "analysis_parameters"),
        expanded=True,
    ):
        st.markdown(
            "\n".join([
                "- " + translate(
                    language,
                    "parameter_sg",
                    window_length=parameters.sg_window_length,
                    polyorder=parameters.sg_polyorder,
                ),
                "- " + translate(
                    language,
                    "parameter_asls",
                    lam=parameters.asls_lam,
                    p=parameters.asls_p,
                ),
                "- " + translate(
                    language,
                    "parameter_peaks",
                    prominence=parameters.peak_prominence,
                    distance=parameters.peak_distance,
                ),
                "- " + translate(language, "parameter_width"),
                "- " + translate(
                    language,
                    "parameter_normalization",
                    method=parameters.normalization_method,
                ),
            ])
        )
        st.caption(translate(language, "peak_detection_note"))


def _create_processing_figure(result: AnalysisResult, language: str):
    spectra = [
        (
            result.raw_spectrum,
            translate(language, "raw_spectrum_title"),
            translate(language, "raw_intensity_axis"),
            "#4C78A8",
        ),
        (
            result.denoised_spectrum,
            translate(language, "denoised_spectrum_title"),
            translate(language, "denoised_intensity_axis"),
            "#F58518",
        ),
        (
            result.corrected_spectrum,
            translate(language, "corrected_spectrum_title"),
            translate(language, "corrected_intensity_axis"),
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
    axes[-1].set_xlabel(translate(language, "raman_shift_axis"))
    return figure


def _create_normalized_figure(result: AnalysisResult, language: str):
    spectrum = result.normalized_spectrum
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(
        spectrum.spectral_axis,
        spectrum.spectral_data,
        color="#B279A2",
        linewidth=1.1,
    )
    axis.set_title(translate(language, "normalized_spectrum_title"))
    axis.set_xlabel(translate(language, "raman_shift_axis"))
    axis.set_ylabel(translate(language, "normalized_intensity_axis"))
    axis.grid(alpha=0.2)
    return figure


def _create_peak_figure(result: AnalysisResult, language: str):
    spectrum = result.corrected_spectrum
    raman_shift = spectrum.spectral_axis
    intensity = spectrum.spectral_data

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(
        raman_shift,
        intensity,
        color="#4C78A8",
        linewidth=1.1,
        label=translate(language, "corrected_line_label"),
    )
    if result.peak_count:
        axis.scatter(
            raman_shift[result.peak_indices],
            intensity[result.peak_indices],
            color="#E45756",
            s=38,
            zorder=3,
            label=translate(language, "candidate_peak_label"),
        )
    axis.set_title(translate(language, "peak_plot_title"))
    axis.set_xlabel(translate(language, "raman_shift_axis"))
    axis.set_ylabel(translate(language, "corrected_intensity_axis"))
    axis.grid(alpha=0.2)
    axis.legend()
    return figure


def _build_peak_table(result: AnalysisResult, language: str) -> pd.DataFrame:
    table = pd.DataFrame.from_records(
        result.peak_features,
        columns=PEAK_TABLE_COLUMNS,
    )
    labels = {
        column: translate(language, label_key)
        for column, label_key in PEAK_TABLE_LABEL_KEYS.items()
    }
    return table.rename(columns=labels)


def _initialize_analysis_state() -> None:
    st.session_state.setdefault(ANALYSIS_RESULT_STATE_KEY, None)
    st.session_state.setdefault(ANALYSIS_REQUEST_STATE_KEY, None)
    st.session_state.setdefault(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
    st.session_state[LANGUAGE_STATE_KEY] = normalize_language(
        st.session_state[LANGUAGE_STATE_KEY]
    )


def _initialize_matching_state() -> None:
    st.session_state.setdefault(MATCHING_RESULT_STATE_KEY, None)
    st.session_state.setdefault(MATCHING_REQUEST_STATE_KEY, None)


def _current_language() -> str:
    return normalize_language(st.session_state.get(LANGUAGE_STATE_KEY))


def _apply_language_selection() -> None:
    old_language = _current_language()
    new_language = normalize_language(
        st.session_state.get(LANGUAGE_SELECTOR_STATE_KEY)
    )
    current_view = st.session_state.get(MAIN_VIEW_STATE_KEY)
    if current_view == translate(old_language, "matching_view"):
        st.session_state[MAIN_VIEW_STATE_KEY] = translate(
            new_language, "matching_view"
        )
    elif current_view == translate(old_language, "analysis_view"):
        st.session_state[MAIN_VIEW_STATE_KEY] = translate(
            new_language, "analysis_view"
        )
    st.session_state[LANGUAGE_STATE_KEY] = new_language
    st.rerun(scope="app")


def _clear_analysis_state() -> None:
    st.session_state[ANALYSIS_RESULT_STATE_KEY] = None
    st.session_state[ANALYSIS_REQUEST_STATE_KEY] = None


def _clear_matching_state() -> None:
    st.session_state[MATCHING_RESULT_STATE_KEY] = None
    st.session_state[MATCHING_REQUEST_STATE_KEY] = None


def _build_analysis_request_key(
    file_bytes: bytes,
    filename: str,
    prominence: object,
) -> tuple[str, str, str]:
    digest = hashlib.sha256(file_bytes).hexdigest()
    prominence_identity = f"{type(prominence).__name__}:{prominence!r}"
    return filename, digest, prominence_identity


def _build_matching_request_key(
    file_bytes: bytes,
    filename: str,
    top_k: int,
) -> tuple[str, str, int]:
    return filename, hashlib.sha256(file_bytes).hexdigest(), top_k


if __name__ == "__main__":
    main()
