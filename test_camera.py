import pandas as pd
import joblib


INPUT_FILE = r"C:\OA SENTINEL\OUTPUT\camera_features.csv"

MODEL_FILE = r"C:\OA SENTINEL\OUTPUT\camera_biomechanics_model.pkl"


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
output1 = r"C:\OA_Sentinel\output\camera_results.csv"
output2 = r"C:\OA SENTINEL\OUTPUT\camera_results.csv"

df.to_csv(output1, index=False)
df.to_csv(output2, index=False)

print("Results saved:")

print(output1)
print(output2)

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
