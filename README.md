# Intelligent Dead Reckoning (IDR) System with GNSS Fusion

**Smart India Hackathon 2026 — AI/ML-Enhanced Sensor Fusion for GNSS-Denied Navigation**

An edge-deployable navigation engine that keeps vehicles accurately tracked using only a smartphone's built-in motion sensors (accelerometer, gyroscope, magnetometer) when GPS/GNSS signal is lost — in tunnels, underground parking, dense urban canyons, or forested highways — and seamlessly re-syncs with GNSS once signal returns.

---

## Problem Statement

GNSS-based navigation apps freeze, jump, or miscalculate routes when signal drops in GNSS-denied environments. Most vehicles on Indian roads (two-wheelers, commercial trucks, older cars) have no factory-fitted INS and rely solely on a dashboard-mounted smartphone. Using a phone's consumer-grade MEMS IMU for dead reckoning is hard — sensor noise, bias, and vibration cause position estimates to drift rapidly and become useless within seconds.

**Goal:** Build a lightweight, edge-deployable engine that transforms a standalone smartphone into an accurate dead-reckoning navigation system, with drift under 10% of distance traveled during GNSS blackout.

---

## System Architecture

```
Accelerometer ─┐
Gyroscope ─────┤ → [ML: clean + flag] → [Math: INS] → [Algorithm: EKF] → Map Matching → Final Position
Magnetometer ──┤                                            ↑
GNSS ──────────┘                                            │
                                                  GNSS fused when available
```

**Important distinction — who does what:**
ML does **not** predict the final speed or position. It only performs two narrow, supporting jobs on the raw sensor window. All actual position/velocity calculation is classical math (INS), and all intelligent blending/correction is a well-defined algorithm (EKF) — not a learned model. This separation matters: it's what makes the system's output explainable and debuggable, rather than a black box.

| Role | Component | What it actually does |
|---|---|---|
| **Clean the input** | ML (ZUPT + bias correction) | Looks at the last 50 sensor readings (~0.5s window) and (a) flags whether the vehicle is currently stationary, (b) optionally corrects systematic sensor noise/bias |
| **Calculate raw motion** | INS (math, not ML) | Integrates the *cleaned* acceleration → velocity → position, and gyroscope → orientation, using standard physics equations |
| **Correct & blend** | EKF (algorithm, not ML) | Takes the INS's raw estimate and intelligently fuses it with GNSS (when available) and map-matching feedback, weighting each source by how much it currently trusts it |
| **Ground truth check** | Map Matching + Road Context | Snaps the EKF's estimate onto real road geometry, and flags implausible speed/heading using OSM road metadata |

### Pipeline Stages (step-by-step data flow)

1. **Sensor Acquisition** — Read live IMU (accelerometer, gyroscope, magnetometer) and GNSS (when available) from the phone, every ~10ms.
2. **Windowing** — Buffer the last 50 readings (~0.5s) into a single window — single readings carry no usable pattern; the model needs short-term context.
3. **ZUPT (Zero-Velocity Update) Detection** — A binary classifier (Random Forest, or small 1D-CNN if more accuracy is needed) looks at the window and outputs: *is the vehicle stationary right now?* (e.g., stopped at a signal). If yes, velocity is forced to zero in the next step — this alone removes a large share of drift.
4. **ML Bias Correction** *(add only if EKF's own bias states aren't enough — see Model Choices below)* — A lightweight on-device model cleans systematic sensor noise/bias from the raw window before it's used in INS.
5. **Strapdown INS** — Classical inertial navigation math (not ML): integrates the cleaned acceleration → velocity → position, and gyroscope → orientation. This is where drift originates if left uncorrected.
6. **Extended Kalman Filter (EKF)** — A 15-state filter (position, velocity, orientation, accelerometer bias, gyroscope bias) fuses the INS's raw estimate with GNSS (when available), weighting each source by real-time confidence. This is the actual "brain" that produces the corrected position — not a learned model, but a well-understood recursive algorithm.
7. **Map Matching** — Snaps the EKF's estimated trajectory onto real road geometry (OpenStreetMap), using non-holonomic constraints (a vehicle can't slide sideways or fly upward).
8. **Road-Context Plausibility Check** — Cross-checks estimated speed/heading against OSM road metadata (speed limits, road type, curvature) to flag and correct implausible drift in real time.
9. **Feedback Loop** — The map-matched position is re-injected into the EKF (step 6) as another correction source, so the EKF's internal belief and the displayed position never diverge.
10. **Navigation UI** — Renders a smooth, uninterrupted vehicle position on the map regardless of GNSS availability, updating at a fixed display rate (~10Hz).

### Model Choices (and why)

| Task | Model | Why |
|---|---|---|
| ZUPT (stationary detection) | Random Forest / Gradient Boosting *(start here)*, or small 1D-CNN if more accuracy is needed | Fast to train, no GPU needed, easy to interpret and tune within a tight timeline. LSTMs are avoided here — detecting "near-zero motion" doesn't need long-term memory. |
| Bias/noise correction | Classical filtering + EKF's own bias states *(try first)* → small 1D-CNN *(only if still needed)* | The EKF already estimates accelerometer/gyro bias as part of its 15-state vector — a separate ML model is only added if that alone doesn't converge well on test data. Avoids redundant complexity. |
| EKF tuning (Q/R covariance) | Not ML — grid search / manual iteration via `filterpy`, validated against ground truth | This is parameter tuning, not model training; it directly determines how much the filter trusts sensors vs. GNSS vs. map-matching at any moment. |

Heavier architectures (LSTM/GRU, Transformers) are deliberately avoided for the on-device models — they add mobile inference cost and tuning time without a clear accuracy benefit for these specific sub-tasks.

---

## Key Features

- **Seamless GNSS ↔ INS switching** — transitions within milliseconds of signal loss/recovery, no manual intervention.
- **No external hardware required** — works with a phone's built-in sensors alone; no OBD-II or vehicle connection needed.
- **In-vehicle alignment & calibration** — automatically determines phone orientation (pitch/roll/yaw) whether dashboard-mounted or handheld.
- **AI-based noise/vibration filtering** — filters out potholes, engine idling vibration, and accidental phone movement.
- **Offline-capable** — core dead-reckoning pipeline and bundled map data work with zero connectivity.
- **Edge-deployable** — models and algorithms also run on external IMU sensor data (not limited to smartphones), per the edge software engine requirement.

---

## Performance Target

| Metric | Benchmark |
|---|---|
| Dead Reckoning Drift | < 10% of distance travelled during GNSS blackout (e.g., < 5m drift over 50m in <1 min, or < 100m drift over 1km at 60km/h) |
| GNSS+INS Fusion Update Rate | 10 Hz (mobile), up to 200 Hz (edge engine with FOG-grade IMU) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model training | Python, PyTorch/TensorFlow, scikit-learn |
| Sensor fusion | `filterpy` (Extended Kalman Filter) |
| Map matching | `leuven-map-matching`, OpenStreetMap (OSM) data |
| On-device inference | TensorFlow Lite / ONNX Runtime Mobile |
| Mobile app | Flutter / React Native |
| Dataset | IO-VNBD (Inertial and Odometry benchmark dataset for ground vehicle positioning) |

---

## Project Structure

```
├── training/
│   ├── data_preprocessing.py      # IO-VNBD loading, windowing, labeling
│   ├── train_bias_correction.py   # ML model: sensor bias/noise correction
│   ├── train_zupt.py              # ML model: zero-velocity detection
│   ├── ekf_tuning.py              # EKF noise parameter (Q/R) tuning
│   └── export_models.py           # Export to TFLite/ONNX
│
├── engine/
│   ├── ins.py                     # Strapdown INS integration
│   ├── ekf.py                     # 15-state Extended Kalman Filter
│   ├── map_matcher.py             # OSM-based map matching
│   └── road_context.py            # Road profile plausibility checks
│
├── mobile_app/
│   ├── sensors/                   # Native IMU/GNSS sensor bridge
│   ├── models/                    # Bundled TFLite models
│   └── ui/                        # Navigation interface
│
├── data/
│   ├── io_vnbd/                   # Dataset (not included — download separately)
│   └── osm_extracts/              # Pre-downloaded map data for offline use
│
└── docs/
    ├── architecture.md            # 2-page architecture document
    └── demo_video_link.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+ (if using React Native) or Flutter SDK
- `pip install -r requirements.txt`

### 1. Download the dataset
Download IO-VNBD and place it under `data/io_vnbd/`.

### 2. Train models
```bash
python training/data_preprocessing.py
python training/train_bias_correction.py
python training/train_zupt.py
python training/ekf_tuning.py
python training/export_models.py
```

### 3. Prepare offline map data
```bash
# Download OSM extract for your test route
python engine/road_context.py --download-region "your_test_area"
```

### 4. Run the mobile app
```bash
cd mobile_app
flutter run   # or: npm run android / npm run ios
```

---

## Evaluation Notes

- Models are trained and validated on the IO-VNBD dataset, with train/test splits performed **by route** (not randomly) to avoid data leakage.
- Live testing simulates GNSS blackout by disabling GPS input during test drives through GNSS-denied environments (tunnels, underground parking).
- Road-context checks use publicly available OpenStreetMap metadata (speed limits, road classification, geometry) as contextual priors — no proprietary or collected traffic data is used.

---

## Team

*[Add team name, members, and institution here]*

## License

*[Add license here]*
