#! /usr/bin/env python
# -*- coding: utf-8 -*-

from . import utils as u
import numpy as np
from numpy import nan_to_num as n2n
from sklearn.metrics import pairwise_distances as smp
import scipy.spatial.distance as ssd
import inspect


def dcos(a, b):
    """ Cosine distance
    """
    
    aa, bb = list(map(np.linalg.norm, [a, b]))
    cos = np.dot(a, b) / (aa * bb)
    
    return cos


def similarity_matrix(a, b, dist='cos', **kwds):
    """ Calculate the similarity matrix of two lists of objects.
    """

    if dist == 'cos':
        pass
    elif dist == 'Euclidean':
        pass
    elif dist == 'Minkowski':
        pass
    elif inspect.isfunction(dist):
        # Calculate with custom-defined distance function
        dist(*[a, b], **kwds)
    else:
        raise NotImplementedError
    
    return


def abserror(result, ref, keys, ofs=None, mask=1, **kwargs):
    """ Calculate the averaged absolute approximation error per band.
    
    :Parameters:
        result : dict
            Dictionary containing the reconstruction results.
        ref : 3d array
            Reference bands or band structure to compare against.
        keys : list/tuple
            Dictionary keys.
        ofs : int | None
            Pixel offset on each side.
        mask : 2d array | 1
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