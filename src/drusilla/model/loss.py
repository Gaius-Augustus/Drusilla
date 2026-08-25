"""Masked categorical cross-entropy loss for ORF label prediction.

Padded positions (where input[..., 5] == 1) must be excluded from the loss.
Class weights compensate for the heavy IR imbalance: START and STOP are
rare but high-value signals, so up-weighting them stabilises training.
"""

from __future__ import annotations

import tensorflow as tf


DEFAULT_CLASS_WEIGHTS = [1.0, 10.0, 1.0, 1.0, 1.0, 10.0]


def masked_crossentropy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    pad_mask: tf.Tensor,
    class_weights: list[float] | None = None,
) -> tf.Tensor:
    """Compute mean cross-entropy, ignoring padded positions.

    Args:
        y_true:       float32 [B, L, 6] one-hot label tensor.
        y_pred:       float32 [B, L, 6] logits.
        pad_mask:     bool    [B, L]    True where position is PAD.
        class_weights: per-class multipliers; defaults to DEFAULT_CLASS_WEIGHTS.
    """
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    weights_t = tf.constant(class_weights, dtype=tf.float32)

    per_pos = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    pos_weight = tf.reduce_sum(y_true * weights_t, axis=-1)

    valid_mask = tf.cast(~pad_mask, tf.float32)
    weighted = per_pos * pos_weight * valid_mask

    n_valid = tf.reduce_sum(valid_mask) + 1e-8
    return tf.reduce_sum(weighted) / n_valid


class MaskedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Keras Loss wrapper around masked_crossentropy.

    Expects y_true to be a float32 tensor of shape [B, L, 7] where the last
    channel encodes the PAD mask (1.0 = padded position).  This packing
    trick lets it work seamlessly with model.compile() and model.fit().
    """

    def __init__(
        self,
        class_weights: list[float] | None = None,
        name: str = "masked_crossentropy",
    ):
        super().__init__(name=name)
        self.class_weights = class_weights or DEFAULT_CLASS_WEIGHTS

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        labels = y_true[..., :6]
        pad_mask = tf.cast(y_true[..., 6], tf.bool)
        return masked_crossentropy(labels, y_pred, pad_mask, self.class_weights)

    def get_config(self):
        cfg = super().get_config()
        cfg["class_weights"] = self.class_weights
        return cfg


class MaskedAccuracy(tf.keras.metrics.Metric):
    """Per-position classification accuracy, ignoring padded positions."""

    def __init__(self, name: str = "accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels   = y_true[..., :6]
        pad_mask = tf.cast(y_true[..., 6], tf.bool)
        valid    = tf.cast(~pad_mask, tf.float32)

        true_cls = tf.argmax(labels, axis=-1)
        pred_cls = tf.argmax(y_pred, axis=-1)
        hits     = tf.cast(tf.equal(true_cls, pred_cls), tf.float32)

        self.correct.assign_add(tf.reduce_sum(hits * valid))
        self.total.assign_add(tf.reduce_sum(valid))

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


CLASS_NAMES = ["IR", "START", "E1", "E2", "E0", "STOP"]


class MaskedF1Score(tf.keras.metrics.Metric):
    """Per-class F1 score at non-padded positions."""

    def __init__(self, class_idx: int, name: str | None = None, **kwargs):
        super().__init__(name=name or f"f1_{CLASS_NAMES[class_idx]}", **kwargs)
        self.class_idx = class_idx
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels   = y_true[..., :6]
        pad_mask = tf.cast(y_true[..., 6], tf.bool)
        valid    = tf.cast(~pad_mask, tf.float32)

        true_cls = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        pred_cls = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        c = tf.cast(self.class_idx, tf.int32)
        true_pos = tf.cast(tf.equal(true_cls, c), tf.float32) * valid
        pred_pos = tf.cast(tf.equal(pred_cls, c), tf.float32) * valid

        self.tp.assign_add(tf.reduce_sum(true_pos * pred_pos))
        self.fp.assign_add(tf.reduce_sum(pred_pos * (1.0 - true_pos)))
        self.fn.assign_add(tf.reduce_sum(true_pos * (1.0 - pred_pos)))

    def result(self):
        p = self.tp / (self.tp + self.fp + 1e-8)
        r = self.tp / (self.tp + self.fn + 1e-8)
        return 2.0 * p * r / (p + r + 1e-8)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


def all_class_f1_metrics() -> list[MaskedF1Score]:
    """Return one MaskedF1Score metric per label class."""
    return [MaskedF1Score(i) for i in range(len(CLASS_NAMES))]


_BOUNDARY_CLASSES = [1, 5]


class MaskedCCEPlusBoundaryF1(tf.keras.losses.Loss):
    """CCE with per-sample class weights + soft-F1 penalty for START and STOP.

    loss = CCE(class_weights) + f1_lambda * mean(1 - soft_F1_c  for c in {START, STOP})
    """

    def __init__(
        self,
        class_weights: list[float] | None = None,
        f1_lambda: float = 1.0,
        name: str = "cce_boundary_f1",
    ):
        super().__init__(name=name)
        self.class_weights = class_weights or DEFAULT_CLASS_WEIGHTS
        self.f1_lambda = f1_lambda

    def _soft_boundary_f1_loss(
        self,
        labels: tf.Tensor,
        probs: tf.Tensor,
        valid: tf.Tensor,
    ) -> tf.Tensor:
        f1_losses = []
        for c in _BOUNDARY_CLASSES:
            p_c = probs[..., c] * valid
            y_c = labels[..., c] * valid
            tp = tf.reduce_sum(p_c * y_c)
            fp = tf.reduce_sum(p_c * (1.0 - y_c))
            fn = tf.reduce_sum((1.0 - p_c) * y_c)
            soft_f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
            f1_losses.append(1.0 - soft_f1)
        return tf.add_n(f1_losses) / float(len(_BOUNDARY_CLASSES))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        labels   = y_true[..., :6]
        pad_mask = tf.cast(y_true[..., 6], tf.bool)
        valid    = tf.cast(~pad_mask, tf.float32)

        cce_term = masked_crossentropy(labels, y_pred, pad_mask, self.class_weights)
        probs    = tf.nn.softmax(y_pred)
        f1_term  = self._soft_boundary_f1_loss(labels, probs, valid)
        return cce_term + self.f1_lambda * f1_term

    def get_config(self):
        cfg = super().get_config()
        cfg["class_weights"] = self.class_weights
        cfg["f1_lambda"]     = self.f1_lambda
        return cfg
