"""
OA SENTINEL — oa_api.py
JointCare Flask Backend for OA Screening
"""

from pathlib import Path
import pandas as pd
from flask import Flask, request, jsonify

from oa_screen import (
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
        "service": "JointCare OA Sentinel API"
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

        # Convert incoming camera data to dataframe
        df = pd.DataFrame(camera_results)

        # Perform screening
        screening = screen_trials(df, reference)

               # Save patient record
        record = append_patient_to_excel(
            patient,
            screening,
            PATIENT_EXCEL
        )

        return jsonify({
            "success": True,
            "patient_id": record["patient_id"],

            # Fields Flutter expects
            "oa_probability": round(100 - record["screening_score"], 1),
            "risk_level": record["screening_level"],
            "knee_stability": round(record["screening_score"], 1),
            "balance_score": round(record["screening_score"], 1),
            "symmetry": round(100 - record["abnormal_trial_rate_pct"], 1),

            # Existing fields
            "screening_level": record["screening_level"],
            "screening_score": record["screening_score"],
            "trials_analyzed": record["trials_analyzed"],
            "mean_biomechanical_score": record["mean_biomechanical_score"],
            "abnormal_trial_rate_pct": record["abnormal_trial_rate_pct"],
            "main_findings": record["main_findings"],
            "note": record["screening_note"]
        }), 200

# -----------------------------
# Run Locally
# -----------------------------
if __name__ == "__main__":
    print("Starting JointCare OA Sentinel API...")
    print("Home   : http://127.0.0.1:5000/")
    print("Health : http://127.0.0.1:5000/health")
    print("Analyze: http://127.0.0.1:5000/analyze")

    app.run(host="0.0.0.0", port=5000, debug=False)
