"""
OA SENTINEL — oa_api.py
JointCare Flask Backend for OA Screening
"""

import logging
import os
import threading
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from oa_screen import (
    FEATURES,
    load_reference,
    screen_trials,
    append_patient_to_excel,
)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# Base paths
BASE = Path(__file__).parent
REFERENCE_FILE = BASE / "oa_healthy_reference.json"
PATIENT_EXCEL = BASE / "OA_Sentinel_Patient_Records.xlsx"
MODEL_FILE = BASE / "camera_biomechanics_model.pkl"

# The Excel log is a local convenience only. Hosts like Render wipe the
# disk on every deploy/restart, so durable records live in the main
# backend's MongoDB. Set SAVE_PATIENT_EXCEL=0 in production.
SAVE_PATIENT_EXCEL = os.environ.get("SAVE_PATIENT_EXCEL", "1") != "0"
EXCEL_LOCK = threading.Lock()

logger = logging.getLogger("oa_api")

# Isolation Forest pipeline (imputer -> scaler -> IsolationForest).
# Loaded once at startup; must match the scikit-learn version it was trained with.
MODEL = joblib.load(MODEL_FILE)


def score_trials(df):
    """
    Run the camera biomechanics model on the incoming trials.

    Any client-supplied biomechanical_prediction / biomechanical_score
    values are overwritten: the server is the source of truth.
    """
    df = df.copy()

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            "camera_results is missing features: " + ", ".join(missing)
        )

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    df["biomechanical_prediction"] = MODEL.predict(X)
    df["biomechanical_score"] = MODEL.decision_function(X)

    if "participant_id" not in df.columns:
        df["participant_id"] = "UNKNOWN"

    return df


def movement_symmetry(df):
    """100 = perfectly symmetric; mean of knee-ROM and step-time asymmetry."""
    asym = pd.concat([
        pd.to_numeric(df["knee_rom_asymmetry_pct"], errors="coerce"),
        pd.to_numeric(df["step_time_asymmetry_pct"], errors="coerce"),
    ]).mean()

    if not np.isfinite(asym):
        return None

    return round(float(np.clip(100.0 - asym, 0.0, 100.0)), 1)

# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "JointCare OA Sentinel API",
        "message": "API is live",
        "version": "1.0"
    }), 200


# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "JointCare OA Sentinel API",
        "model_loaded": MODEL is not None
    }), 200


# -----------------------------
# OA Analysis Endpoint
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({
            "success": False,
            "error": "JSON request body required."
        }), 400

    patient = payload.get("patient", {})
    camera_results = payload.get("camera_results", [])

    if len(camera_results) == 0:
        return jsonify({
            "success": False,
            "error": "camera_results is empty."
        }), 400

    if not REFERENCE_FILE.exists():
        return jsonify({
            "success": False,
            "error": "Healthy reference file not found."
        }), 500

    try:
        # Load healthy reference
        reference = load_reference(REFERENCE_FILE)

        # Convert incoming camera data to dataframe and score it
        df = score_trials(pd.DataFrame(camera_results))

        # Perform screening
        screening = screen_trials(df, reference)

        # Build the patient record; optionally append it to the Excel log.
        # A failed write must not fail the screening itself.
        try:
            with EXCEL_LOCK:
                record = append_patient_to_excel(
                    patient,
                    screening,
                    PATIENT_EXCEL,
                    save=SAVE_PATIENT_EXCEL
                )
        except Exception:
            logger.exception("Could not write patient Excel log")
            record = append_patient_to_excel(
                patient,
                screening,
                PATIENT_EXCEL,
                save=False
            )

        # Return results for Flutter UI
        return jsonify({
            "success": True,
            "patient_id": record["patient_id"],

            # Flutter UI fields.
            # oa_probability is kept as a key for UI compatibility, but it is
            # the 0-100 screening risk index (higher = more unusual gait),
            # NOT a calibrated OA probability.
            "oa_probability": round(record["screening_score"], 1),
            "risk_index": round(record["screening_score"], 1),
            "risk_level": record["screening_level"],
            "symmetry": movement_symmetry(df),
            # Not measured by the camera gait model.
            "knee_stability": None,
            "balance_score": None,

            # Existing API fields
            "screening_level": record["screening_level"],
            "screening_score": record["screening_score"],
            "trials_analyzed": record["trials_analyzed"],
            "mean_biomechanical_score": record["mean_biomechanical_score"],
            "abnormal_trial_rate_pct": record["abnormal_trial_rate_pct"],
            "main_findings": record["main_findings"],
            "note": record["screening_note"]
        }), 200

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -----------------------------
# Run Locally
# -----------------------------
if __name__ == "__main__":
    print("Starting JointCare OA Sentinel API...")
    print("Home   : http://127.0.0.1:5000/")
    print("Health : http://127.0.0.1:5000/health")
    print("Analyze: http://127.0.0.1:5000/analyze")

    app.run(host="0.0.0.0", port=5000, debug=False)
