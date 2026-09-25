import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


# ============================================
# FILES
# ============================================

INPUT_FILE = r"C:\OA SENTINEL\OUTPUT\camera_features.csv"

MODEL_FILE = r"C:\OA SENTINEL\OUTPUT\camera_biomechanics_model.pkl"


# ============================================
# FEATURES
# ============================================

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


# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded")
print("Rows:", len(df))

print()
print("Features:")

for feature in FEATURES:
    print(" -", feature)


# ============================================
# SELECT FEATURES
# ============================================

X = df[FEATURES]


# ============================================
# MODEL
# ============================================

model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42
        )
    )
])


# ============================================
# TRAIN
# ============================================

print()
print("Training camera biomechanics model...")

model.fit(X)


# ============================================
# SAVE
# ============================================

joblib.dump(
    model,
    MODEL_FILE
)


print()
print("Training complete.")

print()
print("Model saved to:")
print(MODEL_FILE)
