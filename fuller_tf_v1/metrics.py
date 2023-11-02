#! /usr/bin/env python
# -*- coding: utf-8 -*-

from . import utils as u
import numpy as np
from numpy import nan_to_num as n2n
# from sklearn.metrics import pairwise_distances as smp
import scipy.spatial.distance as ssd
import itertools as it
import inspect


def dcos(a, b):
    """ Cosine distance between vectors a and b.
    """
    
    aa, bb = list(map(np.linalg.norm, [a, b]))
    cos = np.dot(a, b) / (aa * bb)
    
    return cos


def demean(arr, meanax=1, idx=0, **kwds):
    """ Subtract the mean of an axial direction in an array (2D or higher) from all entries in that direciton.

    **Parameters**\n
    arr: list/tuple/numpy array
        Input array (at least 2D).
    meanax: int | 1
        Axis along which to calculate the mean.
    idx: int | 0
        Entry index in the axis specified previously.
    **kwds: keyword arguments
        Additional arguments for the `numpy.mean()` function.
    """

    arr = np.array(arr)
    arrdm = np.moveaxis(arr, meanax, 0) # demeaned array
    
    mn = np.mean(arrdm[idx,...], **kwds)
    arrdm[idx,...] -= mn
    arrdm = np.moveaxis(arrdm, 0, meanax)
    
    return arrdm


def similarity_matrix(feature_mat, axis=0, fmetric=dcos, **kwds):
    """ Calculation of the similarity matrix.

    **Parameters**\n
    feature_mat: list/tuple/numpy array
        Feature matrix (2D or higher dimenions).
    axis: int
        Axis along which the features are aligned to.
    fmetric: function | dcos
        Metric function for calculating the similarity between each pair of features.
    **kwds: keyword arguments
        Extra arguments for the metric function ``fmetric``.

    **Return**\n
    smat: 2D numpy array
        Calculated similarity matrix.
    """

    if not inspect.isfunction(fmetric):
        raise ValueError('The specified metric should be a function.')
    
    else:
        fmat = np.moveaxis(np.array(feature_mat), axis, 0)
        nfeat = fmat.shape[0]
        smat = np.zeros((nfeat, nfeat))
        ids = list(it.product(range(nfeat), repeat=2))
        
        for pair in ids:
            i, j = pair[0], pair[1]
            smat[i,j] = fmetric(fmat[i,1:], fmat[j,1:], **kwds)
            
        return smat


def abserror(result, ref, keys, ofs=None, mask=1, **kwargs):
    """ Calculate the averaged absolute approximation error per band.
    
    **Parameters**\n
    result: dict
        Dictionary containing the reconstruction results.
    ref: 3D array
        Reference bands or band structure to compare against.
    keys: list/tuple
        Dictionary keys.
    ofs: int | None
        Pixel offset on each side.
    mask: 2D array | 1
        Brillouin zone mask applied to the reconstruction results.
    """
    
    abserr = {}
    outkeys = kwargs.pop('outkeys', keys)
    ret = kwargs.pop('ret', 'dict')
    nnz = np.sum(~np.isnan(mask))
    
    for k, ok in zip(keys, outkeys):
        kstr = str(k)
        okstr = str(ok)
        
        if ofs is not None:
            ofs = int(ofs)
            diffs = mask*(result[kstr][:,ofs:-ofs,ofs:-ofs] - ref)**2
        else:
            diffs = mask*(result[kstr] - ref)**2
        
        diffavgs = np.sqrt(np.sum(n2n(diffs), axis=(1,2)) / nnz)
        abserr[okstr] = diffavgs
    
    if ret == 'dict':
        return abserr
    elif ret == 'array':
        return np.asarray(list(abserr.values()))