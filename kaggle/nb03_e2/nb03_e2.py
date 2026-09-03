"""nb03 -- E2, leave-one-dataset-out (Table II). This is contribution C1.

Train on two datasets, test on a third. No fine-tuning, no target labels, ever.

The headline result is the GAP between E1 and E2, and the plan is emphatic that the
gap is reported honestly even when it is large: concealing it is precisely what the
existing literature is accused of doing. "The gap is larger and harder to close than
the literature implies" is a legitimate, publishable finding.

Adaptation is attempted in ascending order of cost, each rung justified by the
previous being insufficient:
  none          -- the honest baseline
  instance_norm -- per-window normalisation, near-zero MCU cost, start here
  coral         -- aligns second-order statistics, unlabelled target data only
  dann          -- adversarial, highest cost, last resort

Fold D (UMAFall) is the never-trained-on stress test and is reported once. It is also
accelerometer-only -- nb01 established that UMAFall's pocket smartphone logs no
gyroscope and its waist SensorTag samples at 20 Hz -- so fold D additionally carries a
placement shift. That is stated in the table rather than hidden in it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working")

_fdlib_root = next((p.parent for p in Path("/kaggle/input").rglob("fdlib/__init__.py")), None)
if _fdlib_root is None:
    raise SystemExit("fdlib dataset not attached")
sys.path.insert(0, str(_fdlib_root.parent))

from fdlib import config as C  # noqa: E402
from fdlib.adapt import DannSchedule, build_dann, coral_transform  # noqa: E402
from fdlib.cv import class_weights, inner_val_split, lodo_folds  # noqa: E402
from fdlib.experiment import append_row, completed_folds, evaluate, load_corpus  # noqa: E402
from fdlib.models import MacroF1, compile_model, count_params, proposed_cnn, set_seeds  # noqa: E402
from fdlib.preprocess import instance_normalise  # noqa: E402

TASK = "postfall"
CORPUS = next(Path("/kaggle/input").rglob(f"windows_{TASK}.npz"))
CSV = OUT / "results_e2.csv"
METHODS = ("none", "instance_norm", "coral", "dann")

SMOKE = False
if SMOKE:
    C.MAX_EPOCHS, C.EARLY_STOP_PATIENCE = 2, 1

corpus = load_corpus(CORPUS)
X, y = corpus["X"], corpus["y"]
datasets, subjects = corpus["dataset"], corpus["subject"]

print(f"corpus {X.shape}  datasets {sorted(str(d) for d in set(datasets))}")
print(f"proposed params {count_params(proposed_cnn()):,}")
print(f"preprocess signature {C.preprocess_signature()}")


def train_eval(Xtr, ytr, Xva, yva, Xte, yte, method: str, dom_tr=None, n_dom=2):
    """Train the proposed model under one adaptation method and score the unseen set."""
    set_seeds(C.SEED)
    info: dict = {"method": method}

    if method == "instance_norm":
        # Rung 1: normalise each window by its own statistics rather than global
        # training statistics. Costs one pass for mean and one for variance on the MCU.
        Xtr, Xva, Xte = (instance_normalise(a) for a in (Xtr, Xva, Xte))
    elif method == "coral":
        # Rung 2: recolour source features with the TARGET's covariance. Uses target
        # windows but never target labels, so this stays leave-one-dataset-out.
        rng = np.random.default_rng(C.SEED)
        sub_s = rng.choice(len(Xtr), min(3000, len(Xtr)), replace=False)
        sub_t = rng.choice(len(Xte), min(3000, len(Xte)), replace=False)
        try:
            Xtr = np.concatenate([
                coral_transform(Xtr[i:i + 3000], Xte[sub_t])
                for i in range(0, len(Xtr), 3000)
            ])
            info["coral_source_n"] = int(len(sub_s))
        except np.linalg.LinAlgError as e:
            info["coral_error"] = str(e)

    if method == "dann":
        # Rung 3: adversarial. The domain head learns to tell the training datasets
        # apart; gradient reversal makes the trunk unlearn whatever allowed that.
        # Compiled directly rather than through compile_model: that helper passes
        # metrics=["accuracy"], and Keras requires a metrics entry per output on a
        # multi-output model, so routing DANN through it fails at compile time.
        import tensorflow as tf

        base = proposed_cnn()
        model = build_dann(base, n_domains=n_dom)
        steps = max(1, len(Xtr) // C.BATCH_SIZE)
        sched = tf.keras.optimizers.schedules.CosineDecay(
            C.LR, decay_steps=steps * C.MAX_EPOCHS)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=sched),
            loss={"head": "sparse_categorical_crossentropy",
                  "domain": "sparse_categorical_crossentropy"},
            loss_weights={"head": 1.0, "domain": 0.3},
        )
        opt_model = model
        grl = model.get_layer("grl")
        dom_va = np.zeros(len(Xva), np.int64)
        opt_model.fit(
            Xtr, {"head": ytr, "domain": dom_tr},
            validation_data=(Xva, {"head": yva, "domain": dom_va}),
            epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE, verbose=0,
            callbacks=[DannSchedule(grl, C.MAX_EPOCHS)],
        )
        probs = opt_model.predict(Xte, batch_size=512, verbose=0)[0]
        info["params"] = count_params(base)
        return probs, info

    model = compile_model(proposed_cnn(),
                          steps_per_epoch=max(1, len(Xtr) // C.BATCH_SIZE),
                          epochs=C.MAX_EPOCHS)
    cb = MacroF1(Xva, yva)
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=C.MAX_EPOCHS, batch_size=C.BATCH_SIZE, verbose=0,
              class_weight=class_weights(ytr), callbacks=[cb])
    info["params"] = count_params(model)
    info["best_val_macro_f1"] = round(cb.best, 4)
    return model.predict(Xte, batch_size=512, verbose=0), info


print("\n" + "=" * 74)
print("E2 -- leave-one-dataset-out")
print("=" * 74)

for fold_name, tr, te in lodo_folds(datasets):
    # cast away numpy str_ so the CSV carries plain names, not "np.str_(...)"
    train_sets = sorted(str(d) for d in set(datasets[tr]))
    test_set = str(datasets[te][0])
    print(f"\nfold {fold_name}: train {train_sets} -> test {test_set}  "
          f"({len(tr):,} / {len(te):,} windows)")

    inner_tr, inner_va = inner_val_split(subjects, tr)
    dom_map = {d: i for i, d in enumerate(train_sets)}
    dom_tr = np.array([dom_map[str(d)] for d in datasets[inner_tr]], np.int64)

    for method in METHODS:
        key = f"{fold_name}:{method}"
        if key in completed_folds(CSV, "proposed"):
            print(f"  {method}: already done, skipping")
            continue
        t0 = time.time()
        try:
            probs, info = train_eval(
                X[inner_tr], y[inner_tr], X[inner_va], y[inner_va], X[te], y[te],
                method, dom_tr=dom_tr, n_dom=len(train_sets),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {method}: FAILED {type(e).__name__}: {e}")
            continue

        row = {
            "model": "proposed",
            "fold": key,
            "lodo_fold": fold_name,
            "method": method,
            "train_datasets": "+".join(train_sets),
            "test_dataset": str(test_set),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "seconds": round(time.time() - t0, 1),
            "signature": C.preprocess_signature(),
            **evaluate(y[te], probs, info),
        }
        append_row(CSV, row)
        print(f"  {method:14s} macro-F1 {row['macro_f1_3class']:.4f}  "
              f"bin-F1 {row['bin_f1']:.4f}  sens {row['bin_sensitivity']:.3f}  "
              f"spec {row['bin_specificity']:.3f}  ({row['seconds']:.0f}s)")

# ---------------------------------------------------------------------- Table II
df = pd.read_csv(CSV)
piv = df.pivot_table(index=["lodo_fold", "train_datasets", "test_dataset"],
                     columns="method", values="bin_f1").round(4)
piv = piv.reindex(columns=[m for m in METHODS if m in piv.columns])
print("\n" + "=" * 74)
print("TABLE II -- cross-dataset generalisation (F1 on the unseen dataset)")
print("=" * 74)
print(piv.to_string())

piv.to_csv(OUT / "table_II.csv")
notes = (
    "\n**Fold D (UMAFall)** is the never-trained-on stress test, reported once. It is\n"
    "accelerometer-only and at a different body position (pocket, not waist), because\n"
    "UMAFall's waist sensor samples at 20 Hz and its 200 Hz pocket smartphone logs no\n"
    "gyroscope. Fold D therefore measures a combined domain and placement shift and is\n"
    "not directly comparable with folds A-C.\n"
)
(OUT / "table_II.md").write_text(
    "# Table II -- cross-dataset generalisation\n\n"
    "Train on two datasets, test on a third. No fine-tuning, no target labels.\n\n"
    + piv.to_markdown() + "\n" + notes
)
df.to_csv(OUT / "results_e2.csv", index=False)
print("\nwrote results_e2.csv, table_II.csv, table_II.md")
