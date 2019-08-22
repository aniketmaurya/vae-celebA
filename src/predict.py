from vae import VAE
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import yaml


def predict():
    with open("config.yml", 'r') as ymlfile:
        params = yaml.load(ymlfile)

    model_path = '../logs/model-2019-08-22-142947/' + 'vae_weights.h5'
    vae = VAE(params['image_shape'], params['latent_dim'])
    vae.model.load_weights(model_path)
    decoder = vae.model.get_layer('model_1')

    # Display a 2D manifold of the faces
    n = 10  # figure with 15x15 faces
    face_size_i = params['image_shape'][0]
    face_size_j = params['image_shape'][1]
    figure = np.zeros(shape=(face_size_i * n, face_size_j * n))
    # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, params['batch_size']).reshape(params['batch_size'], 2)
            x_decoded = decoder.predict(z_sample, batch_size=params['batch_size'])
            digit = x_decoded[0].reshape(face_size_i, face_size_j)
            figure[i * face_size_i: (i + 1) * face_size_i,
            j * face_size_j: (j + 1) * face_size_j] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    predict()
