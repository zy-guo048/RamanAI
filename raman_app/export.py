"""In-memory export helpers for RamanAI analysis results."""

import csv
from collections.abc import Iterable, Mapping
from io import StringIO


PEAK_FEATURE_COLUMNS = (
    "Peak_ID",
    "Position_cm-1",
    "Intensity",
    "Prominence",
    "Half_Prominence_Width_cm-1",
    "Left_cm-1",
    "Right_cm-1",
    "Width_Height",
)


def peak_features_to_csv(
    peak_features: Iterable[Mapping[str, object]],
) -> bytes:
    """Serialize peak feature records as UTF-8-SIG CSV bytes."""
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PEAK_FEATURE_COLUMNS)
    writer.writeheader()
    for feature in peak_features:
        writer.writerow({column: feature[column] for column in PEAK_FEATURE_COLUMNS})
    return output.getvalue().encode("utf-8-sig")


def build_peak_csv_filename(upload_filename: str) -> str:
    """Build a download name distinguishable from the uploaded TXT file."""
    if upload_filename.casefold().endswith(".txt"):
        stem = upload_filename[:-4]
    else:
        stem = upload_filename
    return f"{stem}_candidate_peak_features.csv"
