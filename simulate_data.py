"""
Simple code to simulate data
"""

# Licence : BSD

import numpy as np
from scipy import linalg, ndimage

from sklearn.utils import check_random_state
from sklearn.feature_selection import f_regression



###############################################################################
# Function to generate data
def create_simulation_data(snr=1., n_samples=200, size=12, roi_size=2,
                           random_state=1):
    generator = check_random_state(random_state)
    smooth_X = 1
    ### Coefs
    w = np.zeros((size, size, size))
    w[0:roi_size, 0:roi_size, 0:roi_size] = -0.6
    w[-roi_size:, -roi_size:, 0:roi_size] = 0.5
    w[0:roi_size, -roi_size:, -roi_size:] = -0.6
    w[-roi_size:, 0:roi_size:, -roi_size:] = 0.5
    w[(size - roi_size) / 2:(size + roi_size) / 2,
      (size - roi_size) / 2:(size + roi_size) / 2,
      (size - roi_size) / 2:(size + roi_size) / 2] = 0.5
    w = w.ravel()
    ### Generate smooth background noise
    XX = generator.randn(n_samples, size, size, size)
    noise = []
    for i in range(n_samples):
        Xi = ndimage.filters.gaussian_filter(XX[i, :, :, :], smooth_X)
        Xi = Xi.ravel()
        noise.append(Xi)
    noise = np.array(noise)
    ### Generate the signal y
    y = generator.randn(n_samples)
    X = np.dot(y[:, np.newaxis], w[np.newaxis])
    norm_noise = linalg.norm(X, 2) / np.exp(snr / 20.)
    noise_coef = norm_noise / linalg.norm(noise, 2)
    noise *= noise_coef
    snr = 20 * np.log(linalg.norm(X, 2) / linalg.norm(noise, 2))
    ### Mixing of signal + noise and splitting into train/test
    X += noise
    X -= X.mean(axis=-1)[:, np.newaxis]
    X /= X.std(axis=-1)[:, np.newaxis]

    return X, y, w, size


def plot_slices(data, title=None):
    pl.figure(figsize=(5.5, 2.2))
    vmax = np.abs(data).max()
    for i in (0, 6, 11):
        pl.subplot(1, 3, i / 5 + 1)
        pl.imshow(data[:, :, i], vmin=-vmax, vmax=vmax,
                  interpolation="nearest", cmap=pl.cm.RdBu_r)
        pl.xticks(())
        pl.yticks(())
    pl.subplots_adjust(hspace=0.05, wspace=0.05, left=.03, right=.97, top=.9)
    if title is not None:
        pl.suptitle(title, y=.95)


if __name__ == '__main__':
    # Create data
    X, y, coefs, size = create_simulation_data(snr=-10, n_samples=100, size=12)

    import pylab as pl
    coefs = np.reshape(coefs, [size, size, size])
    plot_slices(coefs, title="Ground truth")

    f_values, p_values = f_regression(X, y)
    p_values = np.reshape(p_values, (size, size, size))
    p_values = -np.log10(p_values)
    p_values[np.isnan(p_values)] = 0
    p_values[p_values > 10] = 10
    plot_slices(p_values, title="f_regress")

    pl.show()
