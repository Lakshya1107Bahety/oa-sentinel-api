import os
from pathlib import Path

import pandas as pd
import joblib


OUTPUT_DIR = Path(os.environ.get("OA_OUTPUT_DIR", Path(__file__).parent))

INPUT_FILE = OUTPUT_DIR / "camera_features.csv"

MODEL_FILE = OUTPUT_DIR / "camera_biomechanics_model.pkl"


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

    "trunk_lean_deg"
]


# Load
df = pd.read_csv(INPUT_FILE)

model = joblib.load(MODEL_FILE)


# Predict
X = df[FEATURES]

prediction = model.predict(X)

anomaly_score = model.decision_function(X)


# Add results
df["biomechanical_prediction"] = prediction

df["biomechanical_score"] = anomaly_score


# Save
output = OUTPUT_DIR / "camera_results.csv"

df.to_csv(output, index=False)

print("Results saved:")

print(output)

print()

print(
    df[
        [
            "participant_id",
            "trial",
            "biomechanical_prediction",
            "biomechanical_score"
        ]
    ].head(20)
)
