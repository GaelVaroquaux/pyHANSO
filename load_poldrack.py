import numpy as np
import os
from scipy import ndimage

import nibabel as nib


def load_subject_poldrack(subject_id, smooth=0):
    betas_fname = 'data/Jimura_Poldrack_2012_zmaps/gain_realigned/sub0%02d_zmaps.nii.gz' % subject_id
    img = nib.load(betas_fname)
    X = img.get_data()
    affine = img.get_affine()
    finite_mask = np.all(np.isfinite(X), axis=-1)
    mask = np.logical_and(np.all(X != 0, axis=-1),
                            finite_mask)
    if smooth:
        for i in range(X.shape[-1]):
            X[..., i] = ndimage.gaussian_filter(X[..., i], smooth)
        X[np.logical_not(finite_mask)] = np.nan
    y = np.array([np.arange(1, 9)] * 6).ravel()

    assert len(y) == 48
    assert len(y) == X.shape[-1]
    return X, y, mask, affine


poldrack_subjects = np.arange(1, 17)

def load_gain_poldrack(smooth=0):
    X = []
    y = []
    subject = []
    mask = []
    for i in poldrack_subjects:
        X_, y_, this_mask, affine = load_subject_poldrack(i,
                                        smooth=smooth)
        X_ -= X_.mean(axis=-1)[..., np.newaxis]
        std = X_.std(axis=-1)
        std[std==0] = 1
        X_ /= std[..., np.newaxis]
        X.append(X_)
        y.extend(y_)
        subject.extend(len(y_) * [i,])
        mask.append(this_mask)
    X = np.concatenate(X, axis=-1)
    mask = np.sum(mask, axis=0) > .5*len(poldrack_subjects)
    mask = np.logical_and(mask, np.all(np.isfinite(X), axis=-1))
    return X[mask, :].T, np.array(y), np.array(subject), mask, affine

if __name__ == '__main__':
    mask = None
    for i in poldrack_subjects:
        fname = 'data/Jimura_Poldrack_2012_zmaps/gain/sub0%02d_zmaps.nii.gz' % i
        img = nib.load(fname)
        affine = img.get_affine()
        this_mask = np.all(img.get_data() != 0, axis=-1).astype(np.float)
        img = nib.Nifti1Image(this_mask, affine)
        nib.save(img, os.path.join('data/Jimura_Poldrack_2012_zmaps/masks',
                        os.path.basename(fname)))
        if mask is None:
            mask = this_mask
        else:
            mask += this_mask
    mask /= mask.max()
    img = nib.Nifti1Image((mask > .5).astype(np.int), affine)
    nib.save(img, 'data/Jimura_Poldrack_2012_zmaps/mask.nii')

