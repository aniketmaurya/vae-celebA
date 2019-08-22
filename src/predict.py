from vae import VAE
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def predict():
    params = {
        'batch_size': 16,
        'epochs': 2,
        'image_shape': (28, 28, 1),
        'latent_dim': 2,
        'seed': 42
    }
    model_path = '../logs/model-2019-08-22-113322/' + 'vae_weights.h5'
    vae = VAE(params['image_shape'], params['latent_dim'])
    vae.model.load_weights(model_path)
    decoder = vae.model.get_layer('model_1')

    # Display a 2D manifold of the faces
    n = 10  # figure with 15x15 faces
    face_size = params['image_shape'][0]
    figure = np.zeros(shape=(face_size * n, face_size * n))
    # Linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, params['batch_size']).reshape(params['batch_size'], 2)
            x_decoded = decoder.predict(z_sample, batch_size=params['batch_size'])
            digit = x_decoded[0].reshape(face_size, face_size)
            figure[i * face_size: (i + 1) * face_size,
            j * face_size: (j + 1) * face_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    predict()
