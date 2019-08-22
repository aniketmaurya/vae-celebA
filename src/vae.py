from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Lambda, Reshape
from keras import backend as K
from keras.models import Model
from keras.metrics import binary_crossentropy

import numpy as np


class VAE:

    def __init__(self, image_shape, latent_dim):
        self._image_shape = image_shape
        self._latent_dim = latent_dim
        self.model = self._get_vae()

    def _get_z(self):
        """ Encoder network: a very simple convnet which maps the input image x to two vectors,
            z_mean and z_log_variance and then generates a latent space point z.

        """

        input_img = Input(shape=self._image_shape)

        x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
        x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)

        self._shape_before_flattening = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)

        self.z_mean = Dense(self._latent_dim)(x)
        self.z_log_var = Dense(self._latent_dim)(x)

        z = Lambda(VAE.sampling)([self.z_mean, self.z_log_var])

        return input_img, z

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2), mean=0., stddev=1.)

        return z_mean + K.exp(z_log_var) * epsilon

    def _get_decoded_z(self, z):
        decoder_input = Input(K.int_shape(z)[1:])

        # Upsample to the correct number of units
        x = Dense(np.prod(self._shape_before_flattening[1:]), activation='relu')(decoder_input)

        # Reshape into an image of the same shape as before our last Flatten layer
        x = Reshape(self._shape_before_flattening[1:])(x)

        # Apply reverse operation to the initial stacj of Conv2D: a Conv2DTranspose
        x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

        # We end up with a feature map of the same size as the original input
        decoder = Model(decoder_input, x)
        z_decoded = decoder(z)

        return z_decoded

    def _vae_loss(self, input_img, z_decoded):
        input_img = K.flatten(input_img)
        z_decoded = K.flatten(z_decoded)
        xent_loss = binary_crossentropy(input_img, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def _get_vae(self):
        input_img, z = self._get_z()
        z_decoded = self._get_decoded_z(z)

        vae = Model(input_img, z_decoded)
        loss = self._vae_loss(input_img, z_decoded)
        vae.add_loss(loss)
        # Since we have the custom layer we don't specify an external loss at compile time which means also we don't
        # pass target data during training
        vae.compile(optimizer='rmsprop', loss=self._vae_loss)
        vae.summary()

        return vae
