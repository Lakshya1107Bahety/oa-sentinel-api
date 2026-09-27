# oa-sentinel-api

Flask API for OA Sentinel camera-based biomechanical screening.

## Run locally

```bash
pip install -r requirements.txt
python oa_api.py            # http://127.0.0.1:5000
```

Production (e.g. Render): `gunicorn oa_api:app`

## Endpoints

- `GET /health` — status and whether the model loaded
- `POST /analyze` — `{"patient": {...}, "camera_results": [ {10 gait features}, ... ]}`

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SAVE_PATIENT_EXCEL` | `1` | Append each analysis to `OA_Sentinel_Patient_Records.xlsx`. Set `0` on Render: its disk is wiped on every deploy/restart. Durable records are stored by the main backend in MongoDB. |
| `OA_OUTPUT_DIR` | this folder | Where the training scripts read/write CSVs, the model and the reference JSON. |
| `OA_OPENPOSE_ROOT` | `./data/openpose` | Raw OpenPose dataset used by `camera.py`. |

## Retraining pipeline

```bash
python camera.py                        # OpenPose JSON -> camera_features.csv
python train_camera.py                  # -> camera_biomechanics_model.pkl
python test_camera.py                   # -> camera_results.csv
python oa_screen.py --build-reference   # -> oa_healthy_reference.json (run once)
```
