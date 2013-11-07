from time import time

import numpy as np
import nibabel as nib
import pylab as pl

from joblib import Memory
from sklearn.feature_selection import f_regression
from sklearn import linear_model

from load_poldrack import load_gain_poldrack

mem = Memory(cachedir='cache', verbose=3)


if __name__ == '__main__':
    pl.close('all')

    X, y, subjects, mask, affine = mem.cache(load_gain_poldrack)(
                                             smooth=0)

    F, pv = f_regression(X, y)
    data = np.zeros(mask.shape, dtype=np.float32)
    data[mask] = F.astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, 'anova_poldrack.nii')
    del F, pv

    nx, ny, nz = mask.shape
    data = np.zeros((len(X), nx, ny, nz), dtype=np.float32)
    data[:, mask] = X
    del data

    # Fit a Ridge
    ridge = linear_model.RidgeCV()
    ridge.fit(X, y)

    coef_data = np.zeros((nx, ny, nz), dtype=np.float32)
    coef_data[mask] = ridge.coef_.ravel()
    img = nib.Nifti1Image(coef_data, affine)
    nib.save(img, 'ridge_gcv_poldrack_coef.nii')

