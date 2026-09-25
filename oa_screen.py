
"""
OA SENTINEL — oa_screen.py

DO NOT CHANGE:
    camera.py
    train_camera.py
    test_camera.py

This file adds a screening/calibration layer AFTER test_camera.py.

WHAT IT DOES
------------
Your existing Isolation Forest already produces:
    biomechanical_prediction
    biomechanical_score

This script calibrates those outputs against your existing 51-person
healthy/reference dataset.

The final output is:
    LOW
    MODERATE
    HIGH

These are NORMATIVE BIOMECHANICAL SCREENING BANDS.
They are NOT a clinical OA diagnosis and NOT an OA probability.

Reference calibration:
    LOW       <= healthy 90th percentile
    MODERATE  > 90th and <= 97.5th percentile
    HIGH      > healthy 97.5th percentile

The reference must be built ONCE from your existing healthy dataset.
Do not rebuild it after adding new patients.
"""

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd


# ============================================================
# REQUIRED EXISTING COLUMNS
# ============================================================

FEATURES = [
    "right_knee_rom_deg",
    "left_knee_rom_deg",
    "right_hip_rom_deg",
    "left_hip_rom_deg",
    "step_duration_sec",
    "stride_duration_sec",
    "cadence_steps_min",
    "knee_rom_asymmetry_pct",
    "step_time_asymmetry_pct",
    "trunk_lean_deg",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def percentile_rank(values, value):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0 or not np.isfinite(value):
        return np.nan

    return float(
        np.searchsorted(
            np.sort(values),
            value,
            side="right"
        ) / len(values)
    )


def check_columns(df):
    required = [
        "participant_id",
        "biomechanical_prediction",
        "biomechanical_score",
    ] + FEATURES

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "camera_results.csv is missing:\n"
            + "\n".join(missing)
        )


def numeric_clean(df):
    df = df.copy()

    for col in FEATURES + [
        "biomechanical_prediction",
        "biomechanical_score",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# ============================================================
# BUILD HEALTHY REFERENCE — RUN ONCE
# ============================================================

def build_reference(
    healthy_camera_results_csv,
    reference_json
):
    """
    Build patient-level healthy reference statistics from the
    ORIGINAL 51-person reference dataset.

    The reference uses:
      70% = inverse percentile of mean Isolation Forest score
      30% = percentile of abnormal-trial rate

    Higher screening score = more biomechanically unusual.
    """

    df = pd.read_csv(
        healthy_camera_results_csv
    )

    check_columns(df)
    df = numeric_clean(df)

    patient = (
        df.groupby(
            "participant_id",
            as_index=True
        )
        .agg(
            mean_biomechanical_score=(
                "biomechanical_score",
                "mean"
            ),
            abnormal_trial_rate=(
                "biomechanical_prediction",
                lambda x: float(
                    np.mean(
                        np.asarray(x) == -1
                    )
                )
            ),
            trials=(
                "biomechanical_score",
                "count"
            ),
        )
    )

    patient = patient.dropna(
        subset=[
            "mean_biomechanical_score",
            "abnormal_trial_rate",
        ]
    )

    score_values = patient[
        "mean_biomechanical_score"
    ].to_numpy(dtype=float)

    abnormal_values = patient[
        "abnormal_trial_rate"
    ].to_numpy(dtype=float)

    # Existing Isolation Forest:
    # higher score = more normal.
    patient["score_risk_percentile"] = [
        100.0 * (
            1.0 -
            percentile_rank(
                score_values,
                x
            )
        )
        for x in score_values
    ]

    # More anomalous trials = more risk.
    patient["abnormal_rate_percentile"] = [
        100.0 * percentile_rank(
            abnormal_values,
            x
        )
        for x in abnormal_values
    ]

    patient["screening_score"] = (
        0.70 *
        patient["score_risk_percentile"]
        +
        0.30 *
        patient["abnormal_rate_percentile"]
    )

    screening_values = patient[
        "screening_score"
    ].to_numpy(dtype=float)

    reference = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source": str(
            healthy_camera_results_csv
        ),
        "n_reference_participants": int(
            len(patient)
        ),
        "method": {
            "mean_biomechanical_score_weight": 0.70,
            "abnormal_trial_rate_weight": 0.30,
            "low_upper_percentile": 90.0,
            "moderate_upper_percentile": 97.5,
        },
        "thresholds": {
            "p90": float(
                np.percentile(
                    screening_values,
                    90
                )
            ),
            "p97_5": float(
                np.percentile(
                    screening_values,
                    97.5
                )
            ),
        },
        "reference_participants": patient.reset_index()[
            [
                "participant_id",
                "mean_biomechanical_score",
                "abnormal_trial_rate",
                "screening_score",
                "trials",
            ]
        ].to_dict(
            orient="records"
        ),
    }

    Path(reference_json).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(reference_json).write_text(
        json.dumps(
            reference,
            indent=2
        ),
        encoding="utf-8"
    )

    return reference


def load_reference(reference_json):
    path = Path(
        reference_json
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Reference file does not exist: {path}\n"
            "Build it once from the original healthy dataset."
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# TRIAL/PATIENT SCREENING
# ============================================================

def screen_trials(
    camera_results_df,
    reference
):
    """
    Screen one patient's camera_results rows.

    The rows can contain one trial or many trials.
    """

    df = camera_results_df.copy()

    check_columns(df)
    df = numeric_clean(df)

    ref_rows = pd.DataFrame(
        reference[
            "reference_participants"
        ]
    )

    ref_score_values = ref_rows[
        "mean_biomechanical_score"
    ].to_numpy(
        dtype=float
    )

    ref_abnormal_values = ref_rows[
        "abnormal_trial_rate"
    ].to_numpy(
        dtype=float
    )

    mean_score = float(
        df[
            "biomechanical_score"
        ].mean()
    )

    abnormal_rate = float(
        np.mean(
            df[
                "biomechanical_prediction"
            ].to_numpy() == -1
        )
    )

    score_risk = 100.0 * (
        1.0 -
        percentile_rank(
            ref_score_values,
            mean_score
        )
    )

    abnormal_risk = 100.0 * (
        percentile_rank(
            ref_abnormal_values,
            abnormal_rate
        )
    )

    # Final screening index.
    screening_score = (
        0.70 * score_risk
        +
        0.30 * abnormal_risk
    )

    p90 = float(
        reference["thresholds"]["p90"]
    )

    p975 = float(
        reference["thresholds"]["p97_5"]
    )

    if screening_score <= p90:
        level = "LOW"

    elif screening_score <= p975:
        level = "MODERATE"

    else:
        level = "HIGH"

   # --------------------------------------------------------
# Explainability: identify unusual measurements.
# These explanations DO NOT alter the screening score.
# --------------------------------------------------------

# On Render, use the camera_results.csv stored in this repository
reference_csv = Path(__file__).parent / "camera_results.csv"

if not reference_csv.exists():
    raise FileNotFoundError(
        f"camera_results.csv not found at {reference_csv}"
    )

reference_full = pd.read_csv(reference_csv)

reasons = []

    for feature, label in [
        (
            "knee_rom_asymmetry_pct",
            "knee ROM asymmetry"
        ),
        (
            "step_time_asymmetry_pct",
            "step-time asymmetry"
        ),
        (
            "trunk_lean_deg",
            "trunk lean"
        ),
        (
            "cadence_steps_min",
            "cadence"
        ),
        (
            "right_knee_rom_deg",
            "right knee ROM"
        ),
        (
            "left_knee_rom_deg",
            "left knee ROM"
        ),
    ]:

        healthy = pd.to_numeric(
            reference_full[feature],
            errors="coerce"
        ).dropna()

        patient_value = float(
            df[feature].mean()
        )

        if len(healthy) == 0:
            continue

        low = float(
            np.percentile(
                healthy,
                5
            )
        )

        high = float(
            np.percentile(
                healthy,
                95
            )
        )

        if patient_value < low:
            reasons.append(
                f"{label} below healthy 5th percentile"
            )

        elif patient_value > high:
            reasons.append(
                f"{label} above healthy 95th percentile"
            )

    if not reasons:
        reasons.append(
            "No major measured deviation from the reference range"
        )

    summary = {
        "screening_score": round(
            float(screening_score),
            2
        ),
        "screening_level": level,
        "mean_biomechanical_score": round(
            mean_score,
            6
        ),
        "abnormal_trial_rate_pct": round(
            abnormal_rate * 100.0,
            2
        ),
        "score_risk_percentile": round(
            score_risk,
            2
        ),
        "abnormal_rate_percentile": round(
            abnormal_risk,
            2
        ),
        "trials_analyzed": int(
            len(df)
        ),
        "reasons": reasons[:4],
        "threshold_p90": round(
            p90,
            2
        ),
        "threshold_p97_5": round(
            p975,
            2
        ),
        "note": (
            "Normative biomechanical screening only; "
            "not a clinical OA diagnosis or calibrated OA probability."
        ),
    }

    return summary


# ============================================================
# PATIENT EXCEL
# ============================================================

def append_patient_to_excel(
    patient_info,
    screening,
    excel_path
):
    """
    Append one new patient/visit to the Excel file.

    Existing rows are preserved.
    """

    record = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "patient_id": patient_info.get(
            "patient_id",
            "UNKNOWN"
        ),
        "age": patient_info.get(
            "age"
        ),
        "gender": patient_info.get(
            "gender"
        ),
        "height_cm": patient_info.get(
            "height_cm"
        ),
        "mass_kg": patient_info.get(
            "mass_kg"
        ),
        "trials_analyzed": screening[
            "trials_analyzed"
        ],
        "screening_score": screening[
            "screening_score"
        ],
        "screening_level": screening[
            "screening_level"
        ],
        "mean_biomechanical_score": screening[
            "mean_biomechanical_score"
        ],
        "abnormal_trial_rate_pct": screening[
            "abnormal_trial_rate_pct"
        ],
        "score_risk_percentile": screening[
            "score_risk_percentile"
        ],
        "abnormal_rate_percentile": screening[
            "abnormal_rate_percentile"
        ],
        "main_findings": "; ".join(
            screening["reasons"]
        ),
        "threshold_p90": screening[
            "threshold_p90"
        ],
        "threshold_p97_5": screening[
            "threshold_p97_5"
        ],
        "screening_note": screening[
            "note"
        ],
    }

    path = Path(
        excel_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new = pd.DataFrame(
        [record]
    )

    if path.exists():
        old = pd.read_excel(
            path
        )

        combined = pd.concat(
            [old, new],
            ignore_index=True
        )
    else:
        combined = new

    combined.to_excel(
        path,
        index=False
    )

    return record


# ============================================================
# COMMAND-LINE MODE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "OA Sentinel normative biomechanical screening"
        )
    )

    parser.add_argument(
        "--build-reference",
        action="store_true"
    )

    parser.add_argument(
        "--healthy-results",
        default=r"C:\OA SENTINEL\OUTPUT\camera_results.csv"
    )

    parser.add_argument(
        "--patient-results",
        default=r"C:\OA SENTINEL\OUTPUT\camera_results.csv"
    )

    parser.add_argument(
        "--reference",
        default=r"C:\OA SENTINEL\OUTPUT\oa_healthy_reference.json"
    )

    parser.add_argument(
        "--patient-id",
        default="CURRENT_PATIENT"
    )

    parser.add_argument(
        "--patient-excel",
        default=(
            r"C:\OA SENTINEL\OUTPUT"
            r"\OA_Sentinel_Patient_Records.xlsx"
        )
    )

    parser.add_argument(
        "--age",
        type=float,
        default=None
    )

    parser.add_argument(
        "--gender",
        default=None
    )

    parser.add_argument(
        "--height-cm",
        type=float,
        default=None
    )

    parser.add_argument(
        "--mass-kg",
        type=float,
        default=None
    )

    args = parser.parse_args()

    if args.build_reference:

        ref = build_reference(
            args.healthy_results,
            args.reference
        )

        print(
            "\nHealthy reference built."
        )

        print(
            "Participants:",
            ref["n_reference_participants"]
        )

        print(
            "LOW upper threshold:",
            ref["thresholds"]["p90"]
        )

        print(
            "MODERATE upper threshold:",
            ref["thresholds"]["p97_5"]
        )

        raise SystemExit(0)

    reference = load_reference(
        args.reference
    )

    patient_df = pd.read_csv(
        args.patient_results
    )

    screening = screen_trials(
        patient_df,
        reference
    )

    patient_info = {
        "patient_id": args.patient_id,
        "age": args.age,
        "gender": args.gender,
        "height_cm": args.height_cm,
        "mass_kg": args.mass_kg,
    }

    record = append_patient_to_excel(
        patient_info,
        screening,
        args.patient_excel
    )

    trial_output = Path(
        args.patient_results
    ).with_name(
        "oa_screening_results.csv"
    )

    # Add the patient-level result to each output row for convenience.
    patient_output = patient_df.copy()

    patient_output[
        "oa_screening_score"
    ] = screening["screening_score"]

    patient_output[
        "oa_screening_level"
    ] = screening["screening_level"]

    patient_output.to_csv(
        trial_output,
        index=False
    )

    print("\nOA SCREENING COMPLETE")
    print(
        "Patient:",
        args.patient_id
    )
    print(
        "Level:",
        screening["screening_level"]
    )
    print(
        "Screening score:",
        screening["screening_score"]
    )
    print(
        "Findings:",
        "; ".join(screening["reasons"])
    )
    print(
        "Excel:",
        args.patient_excel
    )
