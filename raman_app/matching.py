"""Application adapter for the validated experiment 10 Raman retrieval API."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

from raman_app.pipeline import _validate_filename


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP10_SCRIPT = PROJECT_ROOT / "scripts" / "10_search_rruff_topk.py"
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT
    / "results"
    / "exp09_batch_preprocessing_full"
    / "rruff_preprocessed_300_850_step0p5.npz"
)
DEFAULT_INDEX_PATH = (
    PROJECT_ROOT
    / "results"
    / "exp09_batch_preprocessing_full"
    / "rruff_preprocessing_index.csv"
)
ALLOWED_TOP_K = (3, 5, 10)
REFERENCE_LIBRARY_SOURCE = (
    "experiment_9.3 full preprocessing cache filtered by experiment 10 "
    "processing_status=success and reference_candidate=True"
)
MATCHING_CSV_FIELDS = (
    "query_filename",
    "candidate_rank",
    "reference_label",
    "cosine_similarity",
    "gallery_reference_file_count",
    "gallery_distinct_sample_id_count",
    "best_support_relative_path",
    "best_support_sample_id",
    "best_support_excitation_wavelength_nm",
    "best_minus_second_distinct_sample_id_score",
)


def _load_experiment_10_module():
    module_name = "raman_exp10_topk_for_app"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, EXP10_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the validated experiment 10 retrieval module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp10 = _load_experiment_10_module()


@dataclass(frozen=True)
class SupportingIdentity:
    """Best supporting spectrum for one distinguishable reference identity."""

    rank: int
    identity_basis: str
    identity_key: str
    sample_id: str
    score: float
    relative_path: str
    excitation_wavelength_nm: str
    file_count: int


@dataclass(frozen=True)
class SourceFileFingerprint:
    """Immutable, non-sensitive identity for one loaded reference source file."""

    filename: str
    relative_identifier: str
    sha256: str
    size_bytes: int
    mtime_ns: int | None = None


@dataclass(frozen=True)
class ReferenceLibrarySourceFingerprint:
    """Fingerprint snapshot for the NPZ cache and its reconciled CSV index."""

    npz_cache: SourceFileFingerprint
    index_csv: SourceFileFingerprint


@dataclass(frozen=True)
class ReferenceLibraryBundle:
    """Validated experiment 10 library paired with its source snapshot."""

    library: object
    source_fingerprint: ReferenceLibrarySourceFingerprint


@dataclass(frozen=True)
class MatchingCandidate:
    """One reference-label candidate returned by experiment 10 method A."""

    rank: int
    reference_label: str
    cosine_similarity: float
    reference_file_count: int
    distinguishable_identity_count: int
    best_relative_path: str
    best_sample_id: str
    best_excitation_wavelength_nm: str
    best_minus_second_identity_score: float | None
    supporting_identities: tuple[SupportingIdentity, ...]
    best_reference_vector: np.ndarray


@dataclass(frozen=True)
class SpectrumMatchingResult:
    """UI-ready matching result on the experiment 9.3 engineering grid."""

    filename: str
    query_sha256: str
    requested_top_k: int
    grid: np.ndarray
    query_vector: np.ndarray
    reference_library_size: int
    candidates: tuple[MatchingCandidate, ...]
    source_fingerprint: ReferenceLibrarySourceFingerprint | None = None


def _known_or_unknown(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text if text and text.casefold() not in {"none", "n/a"} else "Unknown"


def _source_relative_identifier(path: Path) -> str:
    resolved = path.resolve()
    try:
        project_relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"filename_only:{resolved.name}"
    return "project_relative:" + project_relative.as_posix()


def _stat_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(
            "Unable to read reference source metadata for "
            f"{path.name}: {type(exc).__name__}"
        ) from exc
    if not path.is_file():
        raise ValueError(f"Reference source is not a file: {path.name}")
    return stat.st_size, stat.st_mtime_ns


def sha256_file(path: Path) -> str:
    """Hash one source file exactly once for a cached library lifecycle."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(
            f"Unable to fingerprint reference source {path.name}: {exc}"
        ) from exc
    return digest.hexdigest()


def _fingerprint_file(path: Path) -> SourceFileFingerprint:
    size_bytes, mtime_ns = _stat_signature(path)
    return SourceFileFingerprint(
        filename=path.name,
        relative_identifier=_source_relative_identifier(path),
        sha256=sha256_file(path),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
    )


def fingerprint_reference_sources(
    cache_path: Path = DEFAULT_CACHE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> ReferenceLibrarySourceFingerprint:
    """Read and hash both source files with surrounding state checks."""
    cache_path = Path(cache_path)
    index_path = Path(index_path)
    before = (_stat_signature(cache_path), _stat_signature(index_path))
    source = ReferenceLibrarySourceFingerprint(
        npz_cache=_fingerprint_file(cache_path),
        index_csv=_fingerprint_file(index_path),
    )
    after = (_stat_signature(cache_path), _stat_signature(index_path))
    if after != before:
        raise ValueError("Reference source changed while fingerprints were computed")
    return source


def _validate_source_fingerprint(
    source: ReferenceLibrarySourceFingerprint | None,
) -> ReferenceLibrarySourceFingerprint:
    if not isinstance(source, ReferenceLibrarySourceFingerprint):
        raise ValueError(
            "Matching result has no validated reference-library source fingerprint"
        )
    for label, item in (("NPZ", source.npz_cache), ("index CSV", source.index_csv)):
        if not item.filename or not item.relative_identifier:
            raise ValueError(f"{label} source fingerprint is incomplete")
        if not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
            raise ValueError(f"{label} source SHA-256 is invalid")
        if (
            not isinstance(item.size_bytes, int)
            or isinstance(item.size_bytes, bool)
            or item.size_bytes < 0
        ):
            raise ValueError(f"{label} source size is invalid")
        if item.mtime_ns is not None and (
            not isinstance(item.mtime_ns, int)
            or isinstance(item.mtime_ns, bool)
            or item.mtime_ns < 0
        ):
            raise ValueError(f"{label} source modification time is invalid")
    return source


def validate_reference_source_snapshot(
    source: ReferenceLibrarySourceFingerprint | None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> None:
    """Reject a snapshot when either currently configured source has changed."""
    source = _validate_source_fingerprint(source)
    for label, item, path in (
        ("NPZ", source.npz_cache, Path(cache_path)),
        ("index CSV", source.index_csv, Path(index_path)),
    ):
        if item.mtime_ns is None:
            raise ValueError(f"{label} reference source snapshot has no file state")
        current_size, current_mtime_ns = _stat_signature(path)
        if path.name != item.filename:
            raise ValueError(f"{label} reference source filename changed")
        if (current_size, current_mtime_ns) != (item.size_bytes, item.mtime_ns):
            raise ValueError(f"{label} reference source changed after matching")


def validate_reference_source_content(
    source: ReferenceLibrarySourceFingerprint | None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> None:
    """Re-hash both sources and reject any content or file-state mismatch."""
    source = _validate_source_fingerprint(source)
    current = fingerprint_reference_sources(cache_path, index_path)
    if current != source:
        raise ValueError(
            "Reference source content no longer matches the matching result"
        )


def _export_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "spectrum"
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return safe or "spectrum"


def build_matching_csv_filename(filename: str) -> str:
    """Return a query-specific filename for the candidate CSV export."""
    return f"{_export_stem(filename)}_raman_matching_candidates.csv"


def build_matching_json_filename(filename: str) -> str:
    """Return a query-specific filename for the retrieval-record JSON export."""
    return f"{_export_stem(filename)}_raman_matching_record.json"


def matching_candidates_to_csv(result: SpectrumMatchingResult) -> bytes:
    """Serialize the currently displayed candidates as UTF-8-SIG CSV."""
    _validate_source_fingerprint(result.source_fingerprint)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MATCHING_CSV_FIELDS)
    writer.writeheader()
    for candidate in result.candidates:
        writer.writerow(
            {
                "query_filename": result.filename,
                "candidate_rank": candidate.rank,
                "reference_label": candidate.reference_label,
                "cosine_similarity": candidate.cosine_similarity,
                "gallery_reference_file_count": candidate.reference_file_count,
                "gallery_distinct_sample_id_count": (
                    candidate.distinguishable_identity_count
                ),
                "best_support_relative_path": candidate.best_relative_path,
                "best_support_sample_id": _known_or_unknown(
                    candidate.best_sample_id
                ),
                "best_support_excitation_wavelength_nm": _known_or_unknown(
                    candidate.best_excitation_wavelength_nm
                ),
                "best_minus_second_distinct_sample_id_score": (
                    candidate.best_minus_second_identity_score
                    if candidate.best_minus_second_identity_score is not None
                    else "Unknown"
                ),
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def _supporting_identity_record(item: SupportingIdentity) -> dict[str, object]:
    return {
        "rank": item.rank,
        "identity_basis": item.identity_basis,
        "identity_key": item.identity_key,
        "sample_id": _known_or_unknown(item.sample_id),
        "maximum_cosine_similarity": item.score,
        "best_support_relative_path": item.relative_path,
        "best_support_excitation_wavelength_nm": _known_or_unknown(
            item.excitation_wavelength_nm
        ),
        "reference_file_count_for_identity": item.file_count,
    }


def matching_record_to_json(result: SpectrumMatchingResult) -> bytes:
    """Serialize one traceable retrieval record without rerunning retrieval."""
    source = _validate_source_fingerprint(result.source_fingerprint)
    candidates = []
    for candidate in result.candidates:
        candidates.append(
            {
                "rank": candidate.rank,
                "reference_label": candidate.reference_label,
                "cosine_similarity": candidate.cosine_similarity,
                "gallery_reference_file_count": candidate.reference_file_count,
                "gallery_distinct_sample_id_count": (
                    candidate.distinguishable_identity_count
                ),
                "best_support_relative_path": candidate.best_relative_path,
                "best_support_sample_id": _known_or_unknown(
                    candidate.best_sample_id
                ),
                "best_support_excitation_wavelength_nm": _known_or_unknown(
                    candidate.best_excitation_wavelength_nm
                ),
                "best_minus_second_distinct_sample_id_score": (
                    candidate.best_minus_second_identity_score
                ),
                "supporting_identities": [
                    _supporting_identity_record(item)
                    for item in candidate.supporting_identities
                ],
            }
        )
    payload = {
        "schema": "ramanai_spectrum_matching_record_v1",
        "query": {
            "filename": result.filename,
            "uploaded_file_sha256": result.query_sha256,
            "sample_id": "Unknown",
            "sample_id_note": "No Sample_ID is inferred from an uploaded filename.",
        },
        "retrieval": {
            "requested_top_k": result.requested_top_k,
            "returned_candidate_count": len(result.candidates),
            "similarity": "cosine_similarity",
            "ranking_level": "reference_label",
            "label_score": "maximum cosine similarity among reference spectra in label",
        },
        "common_grid": {
            "start_cm_1": float(result.grid[0]),
            "end_cm_1": float(result.grid[-1]),
            "step_cm_1": float(result.grid[1] - result.grid[0]),
            "point_count": int(len(result.grid)),
            "extrapolation_allowed": False,
        },
        "preprocessing": {
            "order": [
                "V1 upload parsing",
                "Savitzky-Golay denoising",
                "AsLS baseline correction",
                "max normalization on original sampling axis",
                "linear interpolation to common grid",
            ],
            "sg_window_length_points": 21,
            "sg_polyorder": 3,
            "asls_lam": 1e6,
            "asls_p": 0.01,
            "normalization_method": "max",
            "normalization_stage": "before_interpolation_on_original_axis",
            "interpolation_method": "linear_no_extrapolation",
        },
        "reference_database": {
            "source_identifier": REFERENCE_LIBRARY_SOURCE,
            "actual_reference_spectrum_count": result.reference_library_size,
            "version_hash": None,
            "version_hash_note": (
                "No independent database version identifier is asserted; "
                "source file SHA-256 values are recorded separately."
            ),
            "source_fingerprint": {
                "npz_cache": {
                    "filename": source.npz_cache.filename,
                    "relative_identifier": source.npz_cache.relative_identifier,
                    "sha256": source.npz_cache.sha256,
                    "size_bytes": source.npz_cache.size_bytes,
                },
                "index_csv": {
                    "filename": source.index_csv.filename,
                    "relative_identifier": source.index_csv.relative_identifier,
                    "sha256": source.index_csv.sha256,
                    "size_bytes": source.index_csv.size_bytes,
                },
            },
        },
        "candidates": candidates,
        "scientific_interpretation": (
            "This is a candidate retrieval result against a reference database, "
            "not an externally validated mineral identification conclusion. "
            "Cosine similarity and score gaps are not probability, confidence, "
            "or accuracy."
        ),
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def matching_exports_for_current_source(
    result: SpectrumMatchingResult,
    cache_path: Path = DEFAULT_CACHE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> tuple[bytes, bytes]:
    """Build both downloads only while the result's source snapshot is current."""
    validate_reference_source_content(
        result.source_fingerprint,
        cache_path,
        index_path,
    )
    return matching_candidates_to_csv(result), matching_record_to_json(result)


def matching_result_matches_request(
    result: SpectrumMatchingResult,
    file_bytes: bytes,
    filename: str,
    top_k: int,
) -> bool:
    """Return whether a result belongs to the current upload and Top-K setting."""
    return (
        result.filename == filename
        and result.query_sha256 == hashlib.sha256(file_bytes).hexdigest()
        and result.requested_top_k == top_k
        and len(result.candidates) <= top_k
    )


def load_reference_library(
    cache_path: Path = DEFAULT_CACHE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
):
    """Load and validate the application reference library through experiment 10."""
    return exp10.load_reference_library(Path(cache_path), Path(index_path))


def load_reference_library_with_fingerprint(
    cache_path: Path = DEFAULT_CACHE_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> ReferenceLibraryBundle:
    """Load the exact library sources and freeze their verified fingerprints."""
    cache_path = Path(cache_path)
    index_path = Path(index_path)
    before = (_stat_signature(cache_path), _stat_signature(index_path))
    library = load_reference_library(cache_path, index_path)
    after_load = (_stat_signature(cache_path), _stat_signature(index_path))
    if after_load != before:
        raise ValueError("Reference source changed while the library was loading")
    source_fingerprint = fingerprint_reference_sources(cache_path, index_path)
    after_hash = (_stat_signature(cache_path), _stat_signature(index_path))
    if after_hash != before:
        raise ValueError("Reference source changed while fingerprints were computed")
    return ReferenceLibraryBundle(
        library=library,
        source_fingerprint=source_fingerprint,
    )


def query_from_uploaded_bytes(file_bytes: bytes, filename: str):
    """Apply the exact experiment 9.3 external-RAW flow without a temporary file."""
    _validate_filename(filename)
    try:
        _raw, _corrected, normalized, _dynamic_range = (
            exp10.exp09._run_v1_preprocessing(file_bytes)
        )
    except exp10.exp09.ProcessingFailure as exc:
        raise ValueError(
            "Uploaded RAW preprocessing failed at "
            f"{exc.stage} ({exc.code}): {exc}"
        ) from exc

    source_x = np.asarray(normalized.spectral_axis, dtype=np.float64)
    source_y = np.asarray(normalized.spectral_data, dtype=np.float64)
    target_grid = exp10.exp09.make_trial_grid()
    try:
        exp10.exp09.interpolation.validate_spectrum(
            source_x,
            source_y,
            exp10.exp09.GRID_START_CM_1,
            exp10.exp09.GRID_END_CM_1,
        )
        vector = exp10.exp09.interpolation.linear_interpolate_no_extrapolation(
            source_x,
            source_y,
            target_grid,
        )
    except ValueError as exc:
        raise ValueError(
            "Uploaded RAW cannot cover 300–850 cm-1 without extrapolation: "
            f"{exc}"
        ) from exc

    return exp10.QuerySpectrum(
        vector=np.asarray(vector, dtype=np.float64),
        mode="uploaded_raw_bytes",
        relative_path=f"external:{filename}",
        sample_id="Unknown",
        sample_id_source="missing",
        measurement_stem="Unknown",
        reference_label="Unknown",
        excitation_wavelength_nm="Unknown",
    )


def _supporting_identities(
    similarities: np.ndarray,
    records: tuple[dict[str, str], ...],
    label_key: str,
) -> tuple[SupportingIdentity, ...]:
    grouped: dict[str, dict[str, object]] = {}
    for index, record in enumerate(records):
        if exp10._reference_label_identity(record)[2] != label_key:
            continue
        identity_basis, identity_key = exp10._dedupe_identity(record)
        score = float(similarities[index])
        group = grouped.get(identity_key)
        if group is None:
            grouped[identity_key] = {
                "identity_basis": identity_basis,
                "sample_id": record.get("sample_id", "Unknown") or "Unknown",
                "score": score,
                "record": record,
                "file_count": 1,
            }
            continue
        group["file_count"] = int(group["file_count"]) + 1
        best_record = group["record"]
        assert isinstance(best_record, dict)
        if (
            score > float(group["score"])
            or (
                score == float(group["score"])
                and (record["relative_path"].casefold(), record["relative_path"])
                < (
                    best_record["relative_path"].casefold(),
                    best_record["relative_path"],
                )
            )
        ):
            group["score"] = score
            group["record"] = record

    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -float(item[1]["score"]),
            str(item[1]["sample_id"]).casefold(),
            item[0],
            str(item[1]["record"]["relative_path"]).casefold(),
            str(item[1]["record"]["relative_path"]),
        ),
    )
    output: list[SupportingIdentity] = []
    for rank, (identity_key, group) in enumerate(ordered, start=1):
        record = group["record"]
        assert isinstance(record, dict)
        output.append(
            SupportingIdentity(
                rank=rank,
                identity_basis=str(group["identity_basis"]),
                identity_key=identity_key,
                sample_id=str(group["sample_id"]),
                score=float(group["score"]),
                relative_path=record["relative_path"],
                excitation_wavelength_nm=(
                    record.get("excitation_wavelength_nm", "Unknown") or "Unknown"
                ),
                file_count=int(group["file_count"]),
            )
        )
    return tuple(output)


def match_uploaded_spectrum(
    file_bytes: bytes,
    filename: str,
    k: int,
    library,
    source_fingerprint: ReferenceLibrarySourceFingerprint | None = None,
) -> SpectrumMatchingResult:
    """Match one uploaded RAW spectrum against the experiment 10 application library."""
    if isinstance(k, bool) or k not in ALLOWED_TOP_K:
        raise ValueError("Top-K must be one of 3, 5, or 10")
    query = query_from_uploaded_bytes(file_bytes, filename)
    hits = exp10.search_topk(
        library,
        query,
        k,
        result_level="reference_label",
        exclude_self=False,
        exclude_same_sample=False,
    )
    similarities = exp10.compute_cosine_similarities(
        library.reference_matrix,
        query.vector,
    )
    reference_row_by_path = {
        record["relative_path"]: index
        for index, record in enumerate(library.reference_records)
    }
    if len(reference_row_by_path) != len(library.reference_records):
        raise ValueError("Application reference library contains duplicate paths")

    candidates: list[MatchingCandidate] = []
    for hit in hits:
        record = hit["record"]
        label_key = str(hit["reference_label_group_key"])
        support = _supporting_identities(
            similarities,
            library.reference_records,
            label_key,
        )
        if not support:
            raise ValueError("A returned candidate label has no supporting identity")
        best_index = reference_row_by_path.get(record["relative_path"])
        if best_index is None:
            raise ValueError("Best evidence path is absent from the reference matrix")
        gap = support[0].score - support[1].score if len(support) >= 2 else None
        candidates.append(
            MatchingCandidate(
                rank=int(hit["rank"]),
                reference_label=str(hit["reference_label_group_display"]),
                cosine_similarity=float(hit["cosine_similarity"]),
                reference_file_count=int(hit["label_reference_file_count"]),
                distinguishable_identity_count=int(
                    hit["label_distinguishable_sample_id_count"]
                ),
                best_relative_path=record["relative_path"],
                best_sample_id=record.get("sample_id", "Unknown") or "Unknown",
                best_excitation_wavelength_nm=(
                    record.get("excitation_wavelength_nm", "Unknown") or "Unknown"
                ),
                best_minus_second_identity_score=gap,
                supporting_identities=support[:3],
                best_reference_vector=np.asarray(
                    library.reference_matrix[best_index], dtype=np.float64
                ).copy(),
            )
        )
    if not candidates:
        raise ValueError("No reference-label candidates are available")
    return SpectrumMatchingResult(
        filename=filename,
        query_sha256=hashlib.sha256(file_bytes).hexdigest(),
        requested_top_k=k,
        grid=np.asarray(library.grid, dtype=np.float64).copy(),
        query_vector=np.asarray(query.vector, dtype=np.float64).copy(),
        reference_library_size=len(library.reference_records),
        candidates=tuple(candidates),
        source_fingerprint=source_fingerprint,
    )
