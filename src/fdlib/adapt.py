"""The E2 domain-adaptation ladder, in ascending order of cost.

The plan is explicit that the cross-dataset gap is closed cheapest-first, and that
each rung must be justified by the previous one being insufficient:

  1. per-window instance normalisation -- near-zero MCU cost, often recovers most of
     the loss. Start here.
  2. CORAL feature alignment -- aligns second-order statistics. Needs unlabelled
     target data only, no target labels ever.
  3. DANN -- adversarial. Highest cost, attempted only if 1 and 2 are insufficient.

None of these ever touch target labels. The moment target labels enter, the result
stops being leave-one-dataset-out and the contribution evaporates.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C
from .preprocess import instance_normalise  # noqa: F401  (rung 1, re-exported)


# ------------------------------------------------------------------------- CORAL

def coral_transform(
    source: np.ndarray, target: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Whiten the source features and recolour them with the target's covariance.

    Operates on flattened window features, uses only unlabelled target windows, and
    returns the transformed SOURCE. Nothing is fitted on target labels.
    """
    s = source.reshape(len(source), -1).astype(np.float64)
    t = target.reshape(len(target), -1).astype(np.float64)
    d = s.shape[1]

    ms, mt = s.mean(0), t.mean(0)
    cs = np.cov(s - ms, rowvar=False) + eps * np.eye(d)
    ct = np.cov(t - mt, rowvar=False) + eps * np.eye(d)

    out = ((s - ms) @ _mat_pow(cs, -0.5) @ _mat_pow(ct, 0.5)) + mt
    return out.reshape(source.shape).astype(np.float32)


def _mat_pow(m: np.ndarray, p: float) -> np.ndarray:
    """Symmetric matrix power via eigendecomposition, clipped to stay positive definite."""
    w, v = np.linalg.eigh((m + m.T) / 2.0)
    w = np.clip(w, 1e-10, None)
    return (v * (w ** p)) @ v.T


def coral_loss(source_feat: tf.Tensor, target_feat: tf.Tensor) -> tf.Tensor:
    """Deep CORAL penalty, for use as an auxiliary loss during training."""
    d = tf.cast(tf.shape(source_feat)[1], tf.float32)
    cs = _cov(source_feat)
    ct = _cov(target_feat)
    return tf.reduce_sum(tf.square(cs - ct)) / (4.0 * d * d)


def _cov(x: tf.Tensor) -> tf.Tensor:
    n = tf.cast(tf.shape(x)[0], tf.float32)
    xm = x - tf.reduce_mean(x, axis=0, keepdims=True)
    return tf.matmul(xm, xm, transpose_a=True) / tf.maximum(n - 1.0, 1.0)


# -------------------------------------------------------------------------- DANN

@tf.custom_gradient
def gradient_reversal(x, lam):
    def grad(dy):
        return -lam * dy, None
    return tf.identity(x), grad


class GradientReversal(layers.Layer):
    """Forward identity, backward sign flip -- the whole trick behind DANN."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lam = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="lambda")

    def call(self, x):
        return gradient_reversal(x, self.lam)


def build_dann(base: keras.Model, n_domains: int, feature_layer: str = "gap") -> keras.Model:
    """Attach a domain classifier behind a gradient-reversal layer.

    The label head learns to classify falls; the domain head learns to tell the
    datasets apart; the reversal makes the shared trunk actively unlearn whatever
    made that possible. What survives is dataset-invariant.
    """
    feat = base.get_layer(feature_layer).output
    grl = GradientReversal(name="grl")(feat)
    d = layers.Dense(64, activation="relu", name="dom1")(grl)
    d = layers.Dropout(0.3, name="dom_drop")(d)
    dom = layers.Dense(n_domains, activation="softmax", name="domain")(d)
    return keras.Model(base.input, [base.output, dom], name=f"{base.name}_dann")


class DannSchedule(keras.callbacks.Callback):
    """Ramp lambda from 0 to `max_lam` on the standard DANN schedule.

    Starting at full strength destabilises training: the trunk is asked to be
    domain-invariant before it has learned anything worth being invariant about.
    """

    def __init__(self, layer: GradientReversal, epochs: int, max_lam: float = 1.0):
        super().__init__()
        self.layer, self.epochs, self.max_lam = layer, max(epochs, 1), max_lam

    def on_epoch_begin(self, epoch, logs=None):
        p = epoch / self.epochs
        self.layer.lam.assign(self.max_lam * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


ADAPTATION_METHODS = ("none", "instance_norm", "coral", "dann")
