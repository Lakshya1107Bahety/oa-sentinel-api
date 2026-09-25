"""
OA SENTINEL — oa_api.py
Simple Flask backend.
"""

from pathlib import Path
import pandas as pd
from flask import Flask, request, jsonify

from oa_screen import (
    load_reference,
    screen_trials,
    append_patient_to_excel,
)

app = Flask(__name__)

BASE = Path(__file__).parent
REFERENCE_FILE = BASE / "oa_healthy_reference.json"
PATIENT_EXCEL = BASE / "OA_Sentinel_Patient_Records.xlsx"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "OA Sentinel"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True)

    if not payload:
        return jsonify({
            "success": False,
            "error": "JSON body required"
        }), 400

    patient = payload.get("patient", {})
    camera_results = payload.get("camera_results", [])

    if not camera_results:
        return jsonify({
            "success": False,
            "error": "camera_results is empty"
        }), 400

    if not REFERENCE_FILE.exists():
        return jsonify({
            "success": False,
            "error": "Healthy reference file not found"
        }), 500

    try:
        reference = load_reference(REFERENCE_FILE)
        df = pd.DataFrame(camera_results)
        screening = screen_trials(df, reference)
        record = append_patient_to_excel(patient, screening, PATIENT_EXCEL)

        return jsonify({
            "success": True,
            "patient_id": record["patient_id"],
            "screening_level": record["screening_level"],
            "screening_score": record["screening_score"],
            "trials_analyzed": record["trials_analyzed"],
            "mean_biomechanical_score": record["mean_biomechanical_score"],
            "abnormal_trial_rate_pct": record["abnormal_trial_rate_pct"],
            "main_findings": record["main_findings"],
            "note": record["screening_note"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("Starting OA Sentinel API...")
    print("Health check: http://127.0.0.1:5000/health")
    print("Analysis API: http://127.0.0.1:5000/analyze")
    app.run(host="0.0.0.0", port=5000, debug=False)