# src/models/nn_zero_inflated.py
"""
Definición de la arquitectura NN "zero-inflated":
- embeddings por categoría/ID
- bloque denso profundo con Dropout y BatchNorm
- salida sigmoidal para Und_2a_percentage en [0, 1]
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import LogCosh
from typing import Dict


def build_dynamic_model_tuned(
    embed_cols,
    encoders: Dict[str, object],
    n_numeric_features: int,
    learning_rate: float = 3e-4,
) -> tf.keras.Model:
    """
    Modelo tabular con:
    - Embeddings para cada columna categórica/ID.
    - Bloque denso con capacidad moderada + regularización.
    - Salida sigmoidal adaptada a target continuo [0,1].

    Justificación:
    - Embeddings codifican categorías de alta cardinalidad mejor que One-Hot.
    - LogCosh como loss es más robusto a outliers que MSE puro.
    """
    inputs = []
    embeddings = []

    # 1. Capas de embedding
    for col in embed_cols:
        n_vocab = len(encoders[col].classes_)
        embed_dim = min(60, int(np.log2(n_vocab) * 2.5) + 1)

        in_layer = layers.Input(shape=(1,), name=f"in_{col}")
        inputs.append(in_layer)

        emb = layers.Embedding(input_dim=n_vocab, output_dim=embed_dim)(in_layer)
        emb = layers.Flatten()(emb)
        embeddings.append(emb)

    # 2. Entrada numérica
    if n_numeric_features > 0:
        num_in = layers.Input(shape=(n_numeric_features,), name="in_numerics")
        inputs.append(num_in)
        embeddings.append(num_in)

    # 3. Bloque denso
    x = layers.Concatenate()(embeddings)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)

    # 4. Salida sigmoidal con bias inicial negativo para favorecer valores cercanos a 0
    output = layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(-2.5),
    )(x)

    model = models.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=LogCosh(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model
