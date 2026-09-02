"""Every constant the project depends on.

Nothing here may drift between the training scripts and the firmware generator.
If a value changes, every cached artifact built from it is invalid, which is why
`preprocess_signature()` is stamped into each output file.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------- reproducibility
SEED = 1337

# ---------------------------------------------------------------- signal contract
TARGET_HZ = 50  # working rate after anti-alias + decimate
WINDOW_SEC = 2.0
STRIDE_SEC = 0.5
WINDOW_LEN = int(WINDOW_SEC * TARGET_HZ)  # 100 samples
STRIDE_LEN = int(STRIDE_SEC * TARGET_HZ)  # 50 samples -> 75% overlap

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
N_CHANNELS = len(CHANNELS)

# Native sampling rates of each source, before decimation.
NATIVE_HZ = {
    "sisfall": 200,
    "kfall": 100,
    "fallalld": 238,  # FallAllD waist IMU; confirmed against the release in nb00
    "umafall": 200,   # smartphone waist channel; the 20 Hz dedicated IMU is not used
}

# ---------------------------------------------------------------- class contract
# Three-class head is primary. Binary metrics are derived by merging ALERT+FALL.
BKG, ALERT, FALL = 0, 1, 2
CLASS_NAMES = ("bkg", "alert", "fall")
N_CLASSES = 3

# ---------------------------------------------------------------- training
BATCH_SIZE = 128
MAX_EPOCHS = 100
LR = 1e-3
EARLY_STOP_PATIENCE = 15
DROPOUT = 0.3

# ---------------------------------------------------------------- cross-validation
GROUPED_CV_SPLITS = 5  # subject-grouped, for every comparative result
# Full LOSO is used only for the proposed model's headline number.

# Leave-one-dataset-out folds. UMAFall is test-only in every fold, by design.
LODO_FOLDS = {
    "A": {"train": ["sisfall", "kfall"], "test": "fallalld"},
    "B": {"train": ["sisfall", "fallalld"], "test": "kfall"},
    "C": {"train": ["kfall", "fallalld"], "test": "sisfall"},
    "D": {"train": ["sisfall", "kfall", "fallalld"], "test": "umafall"},
}
HELD_OUT_DATASET = "umafall"

# ---------------------------------------------------------------- evaluation
DECISION_THRESHOLDS = (0.5, 0.7, 0.9)

# ---------------------------------------------------------------- ablations (E5)
ABLATION_WINDOW_SEC = (1.0, 1.5, 2.0)
ABLATION_HZ = (25, 50, 100)

# ---------------------------------------------------------------- deployment budget
MAX_MODEL_KB = 60
MAX_TENSOR_ARENA_KB = 120
MAX_INFERENCE_MS = 50
MAX_END_TO_END_MS = 100
MAX_QUANT_ACCURACY_DROP_PP = 2.0
REPRESENTATIVE_SAMPLES = 200  # drawn from TRAIN only -- drawing from test leaks

# ---------------------------------------------------------------- Kaggle handles
KAGGLE_USER = "arifshekh"
DS_FDLIB = f"{KAGGLE_USER}/fdlib"
DS_WINDOWS = f"{KAGGLE_USER}/fall-windows-50hz"

RAW_SOURCES = {
    "sisfall": "adityavvvn/sisfall",
    "sisfall_enhanced": "nvnikhil0001/sisfall-enhanced",
    "kfall": "usmanabbasi2002/kfall-dataset",
    "fallalld": "sankalpsinghvishen/derived-fallalld-dataset",
    "umafall": "thanushanth/umafall",
}


def preprocess_signature() -> str:
    """Short hash of every value that affects the cached windows.

    Stamped into each .npz so a stale cache can never be silently reused.
    """
    payload = json.dumps(
        {
            "target_hz": TARGET_HZ,
            "window_len": WINDOW_LEN,
            "stride_len": STRIDE_LEN,
            "channels": CHANNELS,
            "native_hz": NATIVE_HZ,
            "n_classes": N_CLASSES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
