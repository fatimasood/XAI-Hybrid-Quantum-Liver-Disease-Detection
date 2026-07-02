#!/usr/bin/env python3

import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses, metrics
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from utils.config import (
    CLASSICAL_LAYERS,
    DROPOUT_RATES,
    LEARNING_RATE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA
)


class ClassicalBaseline:

    def __init__(self, input_dim, learning_rate=LEARNING_RATE):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = None

        self.build_model()
        self.compile_model()

    def build_model(self):

        self.model = Sequential([

            InputLayer(input_shape=(self.input_dim,)),

            Dense(
                CLASSICAL_LAYERS[0],
                activation="relu",
                kernel_initializer="glorot_uniform"
            ),

            Dropout(DROPOUT_RATES[0]),

            Dense(
                CLASSICAL_LAYERS[1],
                activation="relu",
                kernel_initializer="glorot_uniform"
            ),

            Dropout(DROPOUT_RATES[1]),

            Dense(
                64,
                activation="relu",
                kernel_initializer="glorot_uniform"
            ),

            Dense(
                1,
                activation="sigmoid"
            )

        ])

        return self

    def compile_model(self, loss=None):

        if loss is None:
            loss = losses.BinaryFocalCrossentropy(
                gamma=FOCAL_LOSS_GAMMA,
                alpha=FOCAL_LOSS_ALPHA
            )

        self.model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate
            ),
            loss=loss,
            metrics=[
                "accuracy",
                metrics.Precision(name="precision"),
                metrics.Recall(name="recall"),
                metrics.AUC(name="auc")
            ]
        )

        return self

    def get_callbacks(self, fold=None, patience=10):

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.7,
                patience=5,
                min_lr=1e-7
            )

        ]

        if fold is not None:

            os.makedirs("models/saved", exist_ok=True)

            callbacks.append(

                ModelCheckpoint(
                    filepath=f"models/saved/classical_fold_{fold}.h5",
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1
                )

            )

        return callbacks

    def summary(self):

        self.model.summary()

    def save(self, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path)

        print(f"Model saved to {path}")