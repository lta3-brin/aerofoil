import tensorflow as tf


class Aerofoil3BN2FC:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential(
            [
                # Conv Layer 1
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=(9, 9),
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                    input_shape=self.input_shape,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                # Conv Layer 2
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=(7, 7),
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                # Conv Layer 3
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                # Fully connected Layer 1
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                # Fully connected Layer 2
                tf.keras.layers.Dense(3, activation="linear"),
            ]
        )

    def get_model(self):
        return self.model
