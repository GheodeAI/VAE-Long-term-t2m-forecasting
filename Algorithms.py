import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K


def sampling(inputs):
    z_mean = inputs[0]
    z_log_var = inputs[1]
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], 64), mean=0., stddev=1.)
    return z_mean + tf.keras.backend.exp(z_log_var) * epsilon


def random_mask(img):
    import tensorflow as tf
    mask = tf.random.uniform(shape=tf.shape(img)) < .5
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.math.multiply(img, mask)


class AE_PRED:
    """The AE methodology to handle the climate data

    Args:
        input_dim (int): The input size of the image (number of pixels, must be symmetric for both axes)
        latent_dim (int): The size of latent space (number of neurons)
        AE (bool): If True then the variational autoencoder is utilised, otherwise a usual
        channels_in (int): The number of the input variables, e.g., if 'msl', 't2m' and 'sst',
        then channels_in equal to 3
        channels_out (int): The number of the output variables, e.g., if 't2m' only, then channels_out equal 3

    Returns:
        model (h5): The trained model
    """

    def __init__(self, input_dim, latent_dim, AE, channels_in, channels_out):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.AE = AE
        self.channels_in = channels_in
        self.channels_out = channels_out
        self._build()

    def _build(self):
        input_img = keras.Input(shape=(self.input_dim, self.input_dim, self.channels_in))
        x = random_mask(input_img)

        # x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)

        # x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(32, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        # x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(64, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        # x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(128, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(256, 3, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Dropout(0.3)(x)

        encoded = layers.Dense(self.latent_dim, activation=layers.LeakyReLU(alpha=0.3))(x)

        if self.AE:
            z_mean = layers.Dense(self.latent_dim)(encoded)
            z_log_var = layers.Dense(self.latent_dim)(encoded)
            z = tf.keras.layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

            latent_inputs = layers.Input(shape=(self.latent_dim,), name='z_sampling')
            x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(latent_inputs)
        else:
            x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3))(encoded)

        x = layers.Dropout(0.3)(x)
        x = layers.Dense(81, activation=layers.LeakyReLU(alpha=0.3))(x)
        x = layers.Reshape((9, 9, 1))(x)

        x = layers.Conv2DTranspose(128, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)
        x = layers.Conv2DTranspose(64, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=3)(x)
        # x = layers.Conv2DTranspose(32, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)
        # x = layers.Conv2D(3, 4, activation=layers.LeakyReLU(alpha=0.3), padding='same', strides=1)(x)

        decoded = layers.Conv2D(self.channels_out, 2, activation='selu', padding='valid', strides=1)(x)

        if self.AE:
            self.encoder = keras.Model(input_img, [z_mean, z_log_var, z], name='encoder')
            self.decoder = keras.Model(latent_inputs, decoded)
            outputs = self.decoder(self.encoder(input_img)[2])
            self.autoencoder = keras.Model(input_img, outputs)
        else:
            self.encoder = keras.Model(input_img, encoded)
            self.decoder = keras.Model(encoded, decoded)
            self.autoencoder = keras.Model(input_img, self.decoder(self.encoder(input_img)))
        print(self.autoencoder.summary())
        print(self.encoder.summary())

    def compile(self, loss='mse', metrics='mse'):
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)

    def fit(self, x, y, epochs=1000, batch_size=64, shuffle=False, validation_split=0.15, verbose=2, patience=100):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.autoencoder.fit(x, y, epochs=epochs, shuffle=shuffle, validation_split=validation_split,
                             batch_size=batch_size, callbacks=callback, verbose=verbose)

    def encoder(self, x):
        return self.encoder(x)