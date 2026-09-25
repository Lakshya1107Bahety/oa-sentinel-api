import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ============================================================
# SETTINGS
# ============================================================

OPENPOSE_ROOT = Path(r"C:\OA SENTINEL\DATA\ORIGINAL\open pose\Open pose dat")
OUTPUT_DIR = Path(r"C:\OA SENTINEL\OUTPUT")

# OpenPose JSON files are one frame each.
# Change this only if the dataset uses a different frame rate.
FPS = 100.0

# Minimum confidence for a keypoint to be considered reliable
MIN_CONFIDENCE = 0.20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BODY_25 LANDMARK INDICES
# ============================================================

NOSE = 0
NECK = 1
R_SHOULDER = 2
L_SHOULDER = 5
MID_HIP = 8

R_HIP = 9
R_KNEE = 10
R_ANKLE = 11

L_HIP = 12
L_KNEE = 13
L_ANKLE = 14

L_HEEL = 21
R_HEEL = 24


# ============================================================
# READ ONE OPENPOSE JSON
# ============================================================

def read_json(path):

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    people = data.get("people", [])

    if len(people) == 0:
        return None

    # Select the person with the largest number
    # of reliable lower-body landmarks.
    best_person = None
    best_score = -1

    for person in people:

        kp = person.get("pose_keypoints_2d", [])

        if len(kp) < 25 * 3:
            continue

        score = 0

        for idx in [
            MID_HIP,
            R_HIP,
            R_KNEE,
            R_ANKLE,
            L_HIP,
            L_KNEE,
            L_ANKLE,
            L_HEEL,
            R_HEEL
        ]:

            conf = kp[idx * 3 + 2]

            if conf >= MIN_CONFIDENCE:
                score += 1

        if score > best_score:
            best_score = score
            best_person = kp

    if best_person is None:
        return None

    return np.array(best_person, dtype=float).reshape(-1, 3)


# ============================================================
# GET A LANDMARK
# ============================================================

def get_point(kp, index):

    x, y, conf = kp[index]

    if conf < MIN_CONFIDENCE:
        return np.array([np.nan, np.nan])

    return np.array([x, y])


# ============================================================
# ANGLE BETWEEN THREE POINTS
# ============================================================

def angle_3_points(a, b, c):

    # angle ABC

    if np.any(np.isnan(a)) or \
       np.any(np.isnan(b)) or \
       np.any(np.isnan(c)):

        return np.nan

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# ============================================================
# INTERPOLATE MISSING DATA
# ============================================================

def interpolate(values):

    values = np.asarray(values, dtype=float)

    if np.all(np.isnan(values)):
        return values

    series = pd.Series(values)

    series = series.interpolate(
        limit_direction="both"
    )

    return series.to_numpy()


# ============================================================
# SMOOTH SIGNAL
# ============================================================

def smooth(signal, window=7):

    if len(signal) < window:
        return signal

    return pd.Series(signal).rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean().to_numpy()


# ============================================================
# EXTRACT FRAME-LEVEL DATA
# ============================================================

def extract_frames(json_files):

    right_knee_angles = []
    left_knee_angles = []

    right_hip_angles = []
    left_hip_angles = []

    right_heel_x = []
    left_heel_x = []

    right_heel_y = []
    left_heel_y = []

    trunk_angles = []

    valid_frames = 0

    for file in json_files:

        kp = read_json(file)

        if kp is None:
            continue

        # -------------------------
        # Right leg
        # -------------------------

        r_hip = get_point(kp, R_HIP)
        r_knee = get_point(kp, R_KNEE)
        r_ankle = get_point(kp, R_ANKLE)

        # -------------------------
        # Left leg
        # -------------------------

        l_hip = get_point(kp, L_HIP)
        l_knee = get_point(kp, L_KNEE)
        l_ankle = get_point(kp, L_ANKLE)

        # -------------------------
        # Knee angles
        # -------------------------

        r_angle = angle_3_points(
            r_hip,
            r_knee,
            r_ankle
        )

        l_angle = angle_3_points(
            l_hip,
            l_knee,
            l_ankle
        )

        right_knee_angles.append(r_angle)
        left_knee_angles.append(l_angle)

        # -------------------------
        # Hip angle
        #
        # Hip angle is calculated
        # relative to vertical-ish
        # body direction.
        # -------------------------

        mid_hip = get_point(kp, MID_HIP)

        if not np.any(np.isnan(mid_hip)) and \
           not np.any(np.isnan(r_hip)) and \
           not np.any(np.isnan(r_knee)):

            r_hip_angle = angle_3_points(
                mid_hip,
                r_hip,
                r_knee
            )

        else:
            r_hip_angle = np.nan

        if not np.any(np.isnan(mid_hip)) and \
           not np.any(np.isnan(l_hip)) and \
           not np.any(np.isnan(l_knee)):

            l_hip_angle = angle_3_points(
                mid_hip,
                l_hip,
                l_knee
            )

        else:
            l_hip_angle = np.nan

        right_hip_angles.append(r_hip_angle)
        left_hip_angles.append(l_hip_angle)

        # -------------------------
        # Heel positions
        # -------------------------

        r_heel = get_point(kp, R_HEEL)
        l_heel = get_point(kp, L_HEEL)

        right_heel_x.append(r_heel[0])
        right_heel_y.append(r_heel[1])

        left_heel_x.append(l_heel[0])
        left_heel_y.append(l_heel[1])

        # -------------------------
        # Trunk angle
        # -------------------------

        neck = get_point(kp, NECK)

        if not np.any(np.isnan(neck)) and \
           not np.any(np.isnan(mid_hip)):

            dx = neck[0] - mid_hip[0]
            dy = neck[1] - mid_hip[1]

            # angle relative to vertical
            trunk_angle = np.degrees(
                np.arctan2(abs(dx), abs(dy))
            )

        else:
            trunk_angle = np.nan

        trunk_angles.append(trunk_angle)

        valid_frames += 1

    if valid_frames < 20:
        return None

    data = {
        "right_knee_angle": interpolate(right_knee_angles),
        "left_knee_angle": interpolate(left_knee_angles),

        "right_hip_angle": interpolate(right_hip_angles),
        "left_hip_angle": interpolate(left_hip_angles),

        "right_heel_x": interpolate(right_heel_x),
        "left_heel_x": interpolate(left_heel_x),

        "right_heel_y": interpolate(right_heel_y),
        "left_heel_y": interpolate(left_heel_y),

        "trunk_angle": interpolate(trunk_angles)
    }

    return data


# ============================================================
# ROM
# ============================================================

def calculate_rom(signal):

    signal = np.asarray(signal)

    signal = signal[~np.isnan(signal)]

    if len(signal) == 0:
        return np.nan

    return np.percentile(signal, 95) - \
           np.percentile(signal, 5)


# ============================================================
# GAIT TIMING
# ============================================================

def detect_steps(heel_signal):

    signal = np.asarray(heel_signal)

    if len(signal) < 30:
        return []

    signal = smooth(signal, 11)

    # Remove slow movement trend
    t = np.arange(len(signal))

    try:
        coeff = np.polyfit(t, signal, 2)
        trend = np.polyval(coeff, t)
        signal = signal - trend
    except Exception:
        pass

    # Detect repeated movement peaks
    distance = max(
        int(FPS * 0.30),
        1
    )

    prominence = max(
        np.std(signal) * 0.25,
        0.5
    )

    peaks, _ = find_peaks(
        signal,
        distance=distance,
        prominence=prominence
    )

    return peaks


# ============================================================
# GAIT FEATURES
# ============================================================

def calculate_gait_features(
    right_heel_x,
    left_heel_x,
    right_heel_y,
    left_heel_y
):

    right_steps = detect_steps(right_heel_y)
    left_steps = detect_steps(left_heel_y)

    step_times = []

    if len(right_steps) >= 2:

        dt = np.diff(right_steps) / FPS

        step_times.extend(dt.tolist())

    if len(left_steps) >= 2:

        dt = np.diff(left_steps) / FPS

        step_times.extend(dt.tolist())

    if len(step_times) == 0:

        step_duration = np.nan
        stride_duration = np.nan
        cadence = np.nan

    else:

        step_duration = float(
            np.median(step_times)
        )

        stride_duration = step_duration * 2

        cadence = 60.0 / step_duration

    # Step-time asymmetry
    if len(right_steps) >= 2 and \
       len(left_steps) >= 2:

        right_dt = np.median(
            np.diff(right_steps) / FPS
        )

        left_dt = np.median(
            np.diff(left_steps) / FPS
        )

        step_asymmetry = abs(
            right_dt - left_dt
        ) / max(
            (right_dt + left_dt) / 2,
            1e-6
        ) * 100

    else:

        step_asymmetry = np.nan

    return (
        step_duration,
        stride_duration,
        cadence,
        step_asymmetry
    )


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(trial_folder):

    json_files = sorted(
        trial_folder.glob("*_keypoints.json")
    )

    if len(json_files) < 20:
        return None

    # Sort using frame number if present
    def frame_number(path):

        match = re.search(
            r"(\d+)_keypoints\.json$",
            path.name
        )

        if match:
            return int(match.group(1))

        numbers = re.findall(
            r"\d+",
            path.stem
        )

        if numbers:
            return int(numbers[-1])

        return 0

    json_files = sorted(
        json_files,
        key=frame_number
    )

    data = extract_frames(json_files)

    if data is None:
        return None

    # -------------------------
    # Knee ROM
    # -------------------------

    right_knee_rom = calculate_rom(
        data["right_knee_angle"]
    )

    left_knee_rom = calculate_rom(
        data["left_knee_angle"]
    )

    # -------------------------
    # Hip ROM
    # -------------------------

    right_hip_rom = calculate_rom(
        data["right_hip_angle"]
    )

    left_hip_rom = calculate_rom(
        data["left_hip_angle"]
    )

    # -------------------------
    # Knee asymmetry
    # -------------------------

    knee_asymmetry = abs(
        right_knee_rom - left_knee_rom
    ) / max(
        (right_knee_rom + left_knee_rom) / 2,
        1e-6
    ) * 100

    # -------------------------
    # Gait
    # -------------------------

    (
        step_duration,
        stride_duration,
        cadence,
        step_asymmetry
    ) = calculate_gait_features(
        data["right_heel_x"],
        data["left_heel_x"],
        data["right_heel_y"],
        data["left_heel_y"]
    )

    # -------------------------
    # Trunk
    # -------------------------

    trunk_lean = np.nanmedian(
        np.abs(data["trunk_angle"])
    )

    # -------------------------
    # Result
    # -------------------------

    result = {

        "trial": trial_folder.name,

        "n_frames": len(json_files),

        "duration_sec":
            len(json_files) / FPS,

        "right_knee_rom_deg":
            right_knee_rom,

        "left_knee_rom_deg":
            left_knee_rom,

        "right_hip_rom_deg":
            right_hip_rom,

        "left_hip_rom_deg":
            left_hip_rom,

        "step_duration_sec":
            step_duration,

        "stride_duration_sec":
            stride_duration,

        "cadence_steps_min":
            cadence,

        "knee_rom_asymmetry_pct":
            knee_asymmetry,

        "step_time_asymmetry_pct":
            step_asymmetry,

        "trunk_lean_deg":
            trunk_lean
    }

    return result


# ============================================================
# FIND TRIAL FOLDERS
# ============================================================

def find_trial_folders(root):

    folders = []

    for folder in root.rglob("*"):

        if not folder.is_dir():
            continue

        json_files = list(
            folder.glob("*_keypoints.json")
        )

        if len(json_files) >= 20:

            folders.append(folder)

    return folders


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("OA-SENTINEL CAMERA FEATURE EXTRACTION")
    print("=" * 60)

    print()
    print("Searching:")
    print(OPENPOSE_ROOT)

    if not OPENPOSE_ROOT.exists():

        print()
        print("ERROR:")
        print("OpenPose folder does not exist.")
        print()
        print("Create:")
        print(OPENPOSE_ROOT)

        return

    trial_folders = find_trial_folders(
        OPENPOSE_ROOT
    )

    print()
    print(
        f"Found {len(trial_folders)} trial folders."
    )

    if len(trial_folders) == 0:

        print()
        print("No OpenPose trials found.")
        print(
            "Check that your folders contain "
            "*_keypoints.json files."
        )

        return

    results = []

    for i, trial in enumerate(
        trial_folders,
        start=1
    ):

        print(
            f"[{i}/{len(trial_folders)}] "
            f"{trial}"
        )

        try:

            result = process_trial(trial)

            if result is not None:

                # Extract participant number
                numbers = re.findall(
                    r"\d+",
                    str(trial)
                )

                if numbers:

                    result[
                        "participant_id"
                    ] = numbers[0]

                results.append(result)

        except Exception as e:

            print(
                "  ERROR:",
                str(e)
            )

    if len(results) == 0:

        print()
        print(
            "No usable trials were processed."
        )

        return

    df = pd.DataFrame(results)

    # Put participant/trial first
    first_columns = [
        "participant_id",
        "trial",
        "n_frames",
        "duration_sec"
    ]

    remaining = [
        c for c in df.columns
        if c not in first_columns
    ]

    df = df[
        first_columns + remaining
    ]

    output_file = (
        OUTPUT_DIR /
        "camera_features.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print(
        f"Trials processed: {len(df)}"
    )

    print()
    print("Output:")
    print(output_file)

    print()
    print(df.head())


if __name__ == "__main__":
    main()
