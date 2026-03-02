"""
Deep learning model definitions for breathing-pattern classification.

Seven architectures:
  1. LSTM classifier          — stacked LSTM
  2. 1D-CNN classifier        — multi-layer Conv1D
  3. CNN-LSTM hybrid           — Conv1D feature extraction + LSTM temporal context
  4. Bidirectional LSTM        — BiLSTM for forward + backward context
  5. GRU classifier            — lighter recurrent alternative
  6. Attention-LSTM            — LSTM with temporal self-attention
  7. ResNet1D                  — 1D residual network for deep feature extraction

All accept input shape (window_size, n_channels) and produce a single
sigmoid output for binary classification.
"""
from __future__ import annotations

from typing import Optional, Tuple


def _get_keras():
    """Lazy import of tensorflow.keras to keep module importable without TF."""
    try:
        import tensorflow as tf
        return tf.keras
    except ImportError:
        raise ImportError(
            "TensorFlow is required for deep learning models. "
            "Install it with: pip install tensorflow"
        )


# ---------------------------------------------------------------------------
# 1. Stacked LSTM
# ---------------------------------------------------------------------------

def build_lstm_classifier(
    input_shape: Tuple[int, int],
    lstm_units: int = 64,
    lstm_layers: int = 2,
    dense_units: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    keras = _get_keras()
    layers = keras.layers

    model = keras.Sequential(name="LSTM_Classifier")
    model.add(layers.Input(shape=input_shape))

    for i in range(lstm_layers):
        return_seq = i < lstm_layers - 1
        model.add(layers.LSTM(lstm_units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 2. 1D-CNN
# ---------------------------------------------------------------------------

def build_cnn1d_classifier(
    input_shape: Tuple[int, int],
    filters: Tuple[int, ...] = (64, 128, 64),
    kernel_size: int = 5,
    dense_units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    keras = _get_keras()
    layers = keras.layers

    model = keras.Sequential(name="CNN1D_Classifier")
    model.add(layers.Input(shape=input_shape))

    for i, n_filters in enumerate(filters):
        model.add(layers.Conv1D(n_filters, kernel_size, activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        if i < len(filters) - 1:
            model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 3. CNN-LSTM Hybrid
# ---------------------------------------------------------------------------

def build_cnn_lstm_classifier(
    input_shape: Tuple[int, int],
    cnn_filters: Tuple[int, ...] = (64, 128),
    kernel_size: int = 5,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    keras = _get_keras()
    layers = keras.layers

    model = keras.Sequential(name="CNN_LSTM_Classifier")
    model.add(layers.Input(shape=input_shape))

    for n_filters in cnn_filters:
        model.add(layers.Conv1D(n_filters, kernel_size, activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(dropout))

    model.add(layers.LSTM(lstm_units))
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Bidirectional LSTM
# ---------------------------------------------------------------------------

def build_bilstm_classifier(
    input_shape: Tuple[int, int],
    lstm_units: int = 64,
    lstm_layers: int = 2,
    dense_units: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    keras = _get_keras()
    layers = keras.layers

    model = keras.Sequential(name="BiLSTM_Classifier")
    model.add(layers.Input(shape=input_shape))

    for i in range(lstm_layers):
        return_seq = i < lstm_layers - 1
        model.add(layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=return_seq)))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 5. GRU
# ---------------------------------------------------------------------------

def build_gru_classifier(
    input_shape: Tuple[int, int],
    gru_units: int = 64,
    gru_layers: int = 2,
    dense_units: int = 32,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    keras = _get_keras()
    layers = keras.layers

    model = keras.Sequential(name="GRU_Classifier")
    model.add(layers.Input(shape=input_shape))

    for i in range(gru_layers):
        return_seq = i < gru_layers - 1
        model.add(layers.GRU(gru_units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 6. Attention-LSTM (temporal self-attention)
# ---------------------------------------------------------------------------

class _TemporalAttention:
    """Factory for building an LSTM + temporal attention model (Functional API)."""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        lstm_units: int = 64,
        attention_units: int = 32,
        dense_units: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
    ):
        keras = _get_keras()
        layers = keras.layers
        import tensorflow as tf

        inp = layers.Input(shape=input_shape, name="input")

        x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(inp)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_2")(x)
        x = layers.Dropout(dropout)(x)

        # Attention: score each time step, softmax, weighted sum
        score = layers.Dense(attention_units, activation="tanh", name="attn_score")(x)
        alpha = layers.Dense(1, activation="softmax", name="attn_weights")(score)
        # Squeeze last dim for broadcasting, but keep time axis for multiply
        alpha = layers.Reshape((input_shape[0], 1), name="attn_reshape")(alpha)
        context = layers.Multiply(name="attn_context")([x, alpha])
        context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attn_sum")(context)

        x = layers.Dense(dense_units, activation="relu", name="fc1")(context)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = keras.Model(inputs=inp, outputs=out, name="Attention_LSTM_Classifier")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model


def build_attention_lstm_classifier(input_shape, **kwargs):
    return _TemporalAttention.build(input_shape, **kwargs)


# ---------------------------------------------------------------------------
# 7. ResNet1D — residual blocks for deep 1-D feature extraction
# ---------------------------------------------------------------------------

def _resnet1d_block(x, filters, kernel_size, dropout, keras_layers):
    """Single residual block: Conv1D -> BN -> ReLU -> Conv1D -> BN + skip."""
    shortcut = x
    x = keras_layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = keras_layers.BatchNormalization()(x)
    x = keras_layers.Activation("relu")(x)
    x = keras_layers.Dropout(dropout)(x)
    x = keras_layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = keras_layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = keras_layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = keras_layers.BatchNormalization()(shortcut)

    import tensorflow as tf
    x = keras_layers.Add()([x, shortcut])
    x = keras_layers.Activation("relu")(x)
    return x


def build_resnet1d_classifier(
    input_shape: Tuple[int, int],
    block_filters: Tuple[int, ...] = (64, 128, 64),
    kernel_size: int = 5,
    dense_units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    keras = _get_keras()
    layers = keras.layers

    inp = layers.Input(shape=input_shape, name="input")
    x = inp

    for filters in block_filters:
        x = _resnet1d_block(x, filters, kernel_size, dropout, layers)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ResNet1D_Classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_BUILDERS = {
    "LSTM": build_lstm_classifier,
    "CNN1D": build_cnn1d_classifier,
    "CNN_LSTM": build_cnn_lstm_classifier,
    "BiLSTM": build_bilstm_classifier,
    "GRU": build_gru_classifier,
    "Attention_LSTM": build_attention_lstm_classifier,
    "ResNet1D": build_resnet1d_classifier,
}


def get_model(name: str, input_shape: Tuple[int, int], **kwargs):
    """
    Get a compiled model by name.

    Parameters
    ----------
    name : one of the keys in MODEL_BUILDERS
    input_shape : (window_size, n_channels)
    **kwargs : forwarded to the builder function
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[name](input_shape, **kwargs)
