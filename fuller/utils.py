#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import gen_math_ops
from scipy.interpolate import RegularGridInterpolator as RGI
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm_classic
from h5py import File
from silx.io import dictdump
import scipy.io as sio
import natsort as nts
import glob as g
from itertools import product


def nonneg_sum_decomposition(absum, a=None, b=None):
    """ Nonnegative decomposition of a sum.

    Paramters:
        a, b: numeric/None, numeric/None | None, None
            Two numerics for decomposition.
        absum: numeric
            Sum of the values.

    Returns:
        a, b: numeric, numeric
            Nonnegative values of a and b from the decomposition.
    """

    if a is not None:
        if a > absum:
            a = absum
        b = absum - a

        return a, b

    elif b is not None:
        if b > absum:
            b = absum
        a = absum - b

        return a, b

    elif (a is None) and (b is None):
        raise ValueError('At least one of the components should be a numeric.')


def tqdmenv(env):
    """ Choose tqdm progress bar executing environment.

    Parameter:
        env: str
            Name of the environment, 'classic' for ordinary environment,
            'notebook' for Jupyter notebook.
    """

    if env == 'classic':
        tqdm = tqdm_classic
    elif env == 'notebook':
        tqdm = tqdm_notebook

    return tqdm


def to_masked(arr, val=0):
    """ Convert to masked array based on specified value.
    """

    arrm = arr.copy()
    arrm[arrm == val] = np.nan

    return arrm


def valrange(arr):
    """ Output the value range of an array.
    """

    return arr.min(), arr.max()


def interpolate2d(oldx, oldy, vals, nx=None, ny=None, ret='interpolant', **kwargs):
    """ Interpolate values in a newer and/or finer grid.

    **Parameters**\n
    oldx, oldy: 1D array, 1D array
        Values of the old x and y axes.
    vals: 2D array
        Image pixel values associated with the old x and y axes.
    nx, ny: int, int | None, None
        Number of elements in the interpolated axes.
    ret: str | 'interpolant'
        Specification of the return parts.
    **kwargs: keyword arguments
        newx, newy: 1D array, 1D array
            Axes' values after interpolation.
    """

    newx = kwargs.pop('newx', np.linspace(oldx.min(), oldx.max(), nx, endpoint=True))
    newy = kwargs.pop('newy', np.linspace(oldy.min(), oldy.max(), ny, endpoint=True))
    newxymesh = np.meshgrid(newx, newy, indexing='ij')
    newxy = np.stack(newxymesh, axis=-1).reshape((nx*ny, 2))

    vip = RGI((oldx, oldy), vals)
    vals_interp = vip(newxy).reshape((nx, ny))

    if ret == 'interpolant':
        return vals_interp, vip
    elif ret == 'all':
        return vals_interp, vip, newxymesh


def cut_margins(image, margins, offsetx=0, offsety=0):
    """ Trim a 2D image by the given margins.
    """

    offsetx, offsety = int(offsetx), int(offsety)
    yim, xim = image.shape
    t, b, l, r = margins

    if offsetx != 0:
        l, r = l-offsetx, r-offsetx
    if offsety != 0:
        t, b = t-offsety, b-offsety

    image_cut = image[t:yim-b, l:xim-r]

    return image_cut


def findFiles(fdir, fstring='', ftype='h5', **kwds):
    """
    Retrieve files named in a similar way from a folder.
    
    Parameters:
        fdir: str
            Folder name where the files are stored.
        fstring: str | ''
            Extra string in the filename.
        ftype: str | 'h5'
            The type of files to retrieve.
        **kwds: keyword arguments
            Extra keywords for `natsorted()`.
    """
    
    files = nts.natsorted(g.glob(fdir + fstring + '.' + ftype), **kwds)
    
    return files


def saveHDF(*groups, save_addr='./file.h5', track_order=True, **kwds):
    """ Combine dictionaries and save into a hierarchical structure.

    **Parameters**\n
    groups: list/tuple
        Group specified in the following manner that incorporates the name as a string
        and the content and or substructure as a dictionary, ['folder_name', folder_dict].
    save_addr: str | './file.h5'
        File directory for saving the HDF.
    """

    try:
        hdf = File(save_addr, 'w')

        for g in groups:
            grp = hdf.create_group(g[0], track_order=track_order)

            for gk, gv in g[1].items():
                grp.create_dataset(gk, data=gv, **kwds)

    finally:
        hdf.close()


def loadHDF(load_addr, hierarchy='flat', groups='all', track_order=True, dtyp='float', **kwds):
    """ Load contents in an HDF.

    **Parameters**\n
    load_addr: str
        Address of the file to load.
    hierarchy: str | 'flat'
        Hierarchy of the file structure to load into.
    groups: list/tuple/str
        Name of the groups.
    dtype: str | 'float'
        Data type to be loaded into.
    **kwds: keyword arguments
        See ``h5py.File()``.

    **Return**\n
    outdict: dict
        Dictionary containing the hierarchical contents of the file.
    """

    outdict = {}
    if hierarchy == 'nested':
        outdict = dictdump.load(load_addr, fmat='h5')

    elif hierarchy == 'flat':
        with File(load_addr, track_order=track_order, **kwds) as f:

            if groups == 'all':
                groups = list(f)

            for g in groups:
                for gk, gv in f[g].items():
                    outdict[gk] = np.asarray(gv, dtype=dtyp)

    return outdict


def loadH5Parts(filename, content, outtype='dict', alias=None):
    """
    Load specified content from a single complex HDF5 file.
    
    **Parameters**\n
    filename: str
        Namestring of the file.
    content: list/tuple
        Collection of names for the content to retrieve.
    outtype: str | 'dict'
        Option to specify the format of output ('dict', 'list', 'vals').
    alias: list/tuple | None
        Collection of aliases to assign to each entry in content in the output dictionary.
    """
    
    with File(filename) as f:
        if alias is None:
            outdict = {k: np.array(f[k]) for k in content}
        else:
            if len(content) != len(alias):
                raise ValueError('Not every content entry is assigned an alias!')
            else:
                outdict = {ka: np.array(f[k]) for k in content for ka in alias}
    
    if outtype == 'dict':
        return outdict
    elif outtype == 'list':
        return list(outdict.items())
    elif outtype == 'vals':
        return list(outdict.values())


def load_bandstruct(path, form, varnames=[]):
    """ Load band structure information from file.

    **Parameters**\n
    path: str
        File path to load from.
    form: str
        Format of the file to load.
    varnames: list | []
        Names of the variables to load.
    """

    nvars = len(varnames)
    if nvars == 0:
        varnames = ['bands', 'kxx', 'kyy']

    if form == 'mat':
        mat = sio.loadmat(path)
        return [mat[vn] for vn in varnames]

    elif form in ('h5', 'hdf5'):
        dct = loadHDF(path, hierarchy='flat', group=varnames)
        return [dct[vn] for vn in varnames]


def load_multiple_bands(folder, ename='', kname='', form='h5', dtyp='float', **kwargs):
    """ Custom loader for multiple reconstructed bands.

    **Parameters**\n
    folder: str
        Name of the folder.
    ename, kname: str, str | '', ''
        Name of the energy and momentum variables stored in the files.
    form: str | 'h5'
        Format of the files.
    dtype: str | 'float'
        Data type to load the files into.
    **kwargs: keyword arguments
        Extra keywords for ``h5py.File()``.
    """

    if form in ('h5', 'hdf5'):
        files = nts.natsorted(g.glob(f"{folder}/*.h5"))
    else:
        files = nts.natsorted(g.glob(f"{folder}/*.{form}"))

    # Load energy values
    econtents = []
    for f in files:
        f_inst = File(f, **kwargs)
        econtent = np.array(f_inst[ename], dtype=dtyp)
        econtents.append(econtent)

    econtents = np.asarray(econtents)

    # Load momentum values
    kcontents = []
    with f_inst as f_instance:
        kgroups = list(f_instance[kname])

        for kg in kgroups:
            kcontents.append(np.asarray(f_instance[kname][kg], dtype=dtyp))

    return econtents, kcontents


def load_calculation(path, nkx=120, nky=55, delim=' ', drop_pos=2, drop_axis=1, baxis=None, maxid=None):
    """ Read and reshape energy band calculation results.

    **Parameters**\n
    path: str
        File path where the calculation output file is located.
    nkx, nky: int, int
        Number of k points sampled along the kx and ky directions.
    delim: str | ' '
        Delimiter used for reading the calculation output file (default a space string).
    drop_pos, drop_axis: int, int | 2, 1
        The position and axis along which to drop the elements.
    baxis: int | 2
        Axis of the energy band index.
    maxid: int | None
        Maximum limiting index of the read array.

    **Return**\n
    ebands: 3D array
        Collection of energy bands indexed by their energies.
    """

    nkx, nky = int(nkx), int(nky)
    nk = nkx*nky
    arr = np.fromfile(path, sep=delim)
    neb = int(arr.size / nk)

    if maxid is None:
        ebands = arr[:nk*neb].reshape((nk, neb))
    else:
        maxid = int(maxid)
        ebands = arr[:maxid].reshape((nk, neb))

    if drop_axis is not None: # Drop the constant column (i.e. the kz axis)
        ebands = np.delete(ebands, drop_pos, axis=drop_axis).reshape((nky, nkx, neb-1))

    if baxis is not None:
        baxis = int(baxis)
        ebands = np.moveaxis(ebands, 2, baxis)

    return ebands


def pick_operator(fstring, package='numpy'):
    """ Return an operator function from the specified pacakge.

    Parameter:
        sstring: str
            The namestring of the numpy function.
        package: str | 'numpy'
            The name of the software package to extract the function.
    """

    try:
        exec('import ' + package)
        return eval(package + '.' + fstring)
    except:
        return fstring


def nzbound(arr):
    """ Find index bounds of the nonzero elements of a 1D array.
    """

    arr = np.asarray(arr)
    axis_nz_index = np.argwhere(arr!=0).ravel()

    return axis_nz_index[0], axis_nz_index[-1]


def segmod(indices):
    """ Add 1 to the intermediate indices.
    """

    alt_indices = indices + 1
    alt_indices[0] -= 1
    alt_indices[-1] -= 1

    return alt_indices


def fexp(ke, length):
    """ Exponential function.
    """
    
    return np.exp(-ke * np.arange(0, length, 1))
    

def coeffgen(size, amp=1, distribution='uniform', mask=None, modulation=None, seed=None, **kwargs):
    """ Generate random sequence from a distribution modulated by an envelope function and a mask.
    
    **Parameters**\n
    size: list/tuple
        Size of the coefficient array.
    amp: numeric | 1
        Global amplitude scaling of the random sequence.
    distribution: str | 'uniform'
        Type of distribution to draw from.
    mask: ndarray | None
        Amplitude mask array.
    modulation: ndarray/str | None
        Amplitude modulation array.
    seed: numeric | None:
        Seed value for the random number generator.
    **kwargs: keyword arguments
        Additional arguments for the specified distribution function.s
    """
    
    op_package = kwargs.pop('package', 'numpy.random')
    
    # Seeding random number generation
    if seed is not None:
        np.random.seed(seed)
    
    # Apply envelope modulation
    if modulation is not None:
        if modulation == 'exp':
            ke = kwargs.pop('ke', 2e-2)
            length = kwargs.pop('length', size[1])
            cfmod = fexp(ke, length)[None, :]
        elif type(modulation) == np.ndarray:
            cfmod = modulation
    else:
        cfmod = np.ones(size)
    
    # Apply zero mask
    if mask is not None:
        if mask.ndim == 1:
            cfmask = mask[None, :]
        elif type(mask) == np.ndarray:
            cfmask = mask
    else:
        cfmask = np.ones(size)
        
    # Generate basis coefficient
    opr = pick_operator(distribution, package=op_package)
    cfout = opr(size=size, **kwargs)
    
    cfout *= amp*cfmask*cfmod
    
    return cfout


def binarize(cfs, threshold, vals=[0, 1], absolute=True, eq='geq'):
    """ Binarize an array by a threshold.

    **Parameters**\n
    cfs: list/tuple/numpy array
        Numerical object.
    threshold: numeric
        Numerical threshold for binarization.
    vals: list/tuple/numpy array
        Values assigned to the two sides of the threshold.
    absolute: bool | True
        Option to use the absolute value for thresholding.
    eq: str | 'geq'
        Options to treat the values equal to the threshold (`'leq'` for less or equal,
        `'geq'` for greater or equal, `None` for drop the threshold-equalling values).

    **Return**\n
    arr: list/tuple/numpy array
        Binarized array.
    """
    
    arr = np.array(cfs)
    if absolute:
        arr = np.abs(arr)
    
    if eq == 'leq':
        arr[arr <= threshold] = vals[0]
        arr[arr > threshold] = vals[1]
    elif eq == 'geq':
        arr[arr < threshold] = vals[0]
        arr[arr >= threshold] = vals[1]
    elif eq is None:
        arr[arr < threshold] = vals[0]
        arr[arr > threshold] = vals[1]
    
    return arr


def trim_2d_edge(arr, edges, axes=(0, 1)):
    """ Trim 2D edges in the first two dimensions of an nD array.

    **Parameters**\n
    arr: numpy array
        Array to trim .
    edges: numeric/list/tuple/numpy array
        The amount of edges to trim. If a single value is assigned, the two ends of the
        axes are trimmed equally. If a list of four different values is assigned, they are
        applied to the two axes in the order `(start_1, end_1, start_2, end_2)`.
    axes: list/tuple
        Specified axes/dimensions to trim.

    **Return**\n
    trimmed: numpy array
        Axis-trimmed array.
    """
    
    edges = np.array(edges)
    trimmed = np.moveaxis(arr, axes, (0, 1))
    
    if edges.size == 1:
        eg = edges.item()
        trimmed = trimmed[eg:-eg,eg:-eg,...]
    
    elif edges.size == 4:
        top, bot, left, rite = edges
        trimmed = trimmed[top:-bot, left:-rite,...]

    trimmed = np.moveaxis(trimmed, (0, 1), axes)
    
    return trimmed