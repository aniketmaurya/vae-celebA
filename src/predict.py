from vae import VAE
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import yaml


def predict():
    with open("config.yml", 'r') as ymlfile:
        params = yaml.safe_load(ymlfile)

    model_path = '../logs/model-2019-08-22-162729/' + 'vae_weights.h5'
    vae = VAE(params['image_shape'], params['latent_dim'])
    vae.model.load_weights(model_path)
    decoder = vae.model.get_layer('model_1')

    # Use the decoder network to turn arbitrary latent space vectors into images and display a 2D manifold of the faces
    n = 5  # figure with 15x15 faces
    face_size_i = params['image_shape'][0]
    face_size_j = params['image_shape'][1]
    color_channel = params['image_shape'][2]
    figure = np.zeros(shape=(face_size_i * n, face_size_j * n, color_channel))
    # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.55, 0.85, n))
    grid_y = norm.ppf(np.linspace(0.35, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, params['batch_size']).reshape(params['batch_size'], 2)
            x_decoded = decoder.predict(z_sample, batch_size=params['batch_size'])
            face = x_decoded[0].reshape(face_size_i, face_size_j, color_channel)
            figure[i * face_size_i: (i + 1) * face_size_i,
            j * face_size_j: (j + 1) * face_size_j] = face
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    predict()
