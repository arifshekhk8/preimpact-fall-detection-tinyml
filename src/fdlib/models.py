"""Model definitions. The MCU memory budget is the design driver, not an afterthought.

The proposed model targets ~25,000 parameters, which quantises to roughly 25 KB --
comfortably inside the ESP32's flash and, more importantly, inside a tensor arena
that leaves room for the rest of the firmware.

Two choices carry that budget:
  * separable convolutions, which cut parameter count by roughly an order of
    magnitude versus standard convolutions at similar accuracy;
  * global average pooling instead of flatten, which stops the dense layer from
    dominating the budget -- a flatten here would cost 25*64*32 = 51k parameters in
    one layer, twice the entire budget.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


class InstanceNorm(layers.Layer):
    """Per-window, per-channel normalisation, inside the graph.

    Keeping it in the graph rather than in the data pipeline means the exported
    TFLite model carries it, so the firmware does not have to reproduce a second
    normalisation scheme by hand -- one less place for train/deploy skew to hide.
    """

    def __init__(self, eps: float = 1e-6, **kw):
        super().__init__(**kw)
        self.eps = eps

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps)

    def get_config(self):
        return {**super().get_config(), "eps": self.eps}


def proposed_cnn(
    input_len: int = C.WINDOW_LEN,
    n_channels: int = C.N_CHANNELS,
    n_classes: int = C.N_CLASSES,
    instance_norm: bool = True,
) -> keras.Model:
    """The compact separable CNN of section 6."""
    inp = keras.Input(shape=(input_len, n_channels), name="window")
    x = InstanceNorm(name="instance_norm")(inp) if instance_norm else inp

    x = layers.Conv1D(24, 7, strides=2, padding="same", use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    x = layers.SeparableConv1D(48, 5, padding="same", use_bias=False, name="sep1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(2, name="pool1")(x)

    x = layers.SeparableConv1D(64, 3, padding="same", use_bias=False, name="sep2")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(C.DROPOUT, name="drop")(x)
    out = layers.Dense(n_classes, activation="softmax", name="head")(x)
    return keras.Model(inp, out, name="proposed_separable_cnn")


def cnn_1d(
    input_len: int = C.WINDOW_LEN,
    n_channels: int = C.N_CHANNELS,
    n_classes: int = C.N_CLASSES,
) -> keras.Model:
    """Table I comparator: a conventional 1D CNN, no separable trick, no size budget."""
    inp = keras.Input(shape=(input_len, n_channels))
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(C.DROPOUT)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_1d")


def cnn_lstm(
    input_len: int = C.WINDOW_LEN,
    n_channels: int = C.N_CHANNELS,
    n_classes: int = C.N_CLASSES,
) -> keras.Model:
    """Table I comparator: the CNN-LSTM that dominates this literature."""
    inp = keras.Input(shape=(input_len, n_channels))
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(C.DROPOUT)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_lstm")


MODELS = {
    "proposed": proposed_cnn,
    "cnn_1d": cnn_1d,
    "cnn_lstm": cnn_lstm,
}


def compile_model(model: keras.Model, lr: float = C.LR, steps_per_epoch: int | None = None,
                  epochs: int = C.MAX_EPOCHS) -> keras.Model:
    """Adam with cosine decay, as specified."""
    if steps_per_epoch:
        sched = keras.optimizers.schedules.CosineDecay(lr, decay_steps=steps_per_epoch * epochs)
        opt = keras.optimizers.Adam(learning_rate=sched)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


class MacroF1(keras.callbacks.Callback):
    """Early stopping on validation macro-F1, not accuracy.

    Accuracy is useless here: background is ~97% of windows, so a model that never
    fires scores 0.97 and stops improving on the only thing that matters.
    """

    def __init__(self, val_x, val_y, patience: int = C.EARLY_STOP_PATIENCE):
        super().__init__()
        self.val_x, self.val_y, self.patience = val_x, val_y, patience
        self.best, self.wait, self.best_weights = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        from .metrics import macro_f1_3class
        p = self.model.predict(self.val_x, verbose=0, batch_size=512)
        f1 = macro_f1_3class(self.val_y, p)
        (logs if logs is not None else {})["val_macro_f1"] = f1
        if f1 > self.best:
            self.best, self.wait = f1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def count_params(model: keras.Model) -> int:
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def set_seeds(seed: int = C.SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
