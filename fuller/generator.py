#! /usr/bin/env python
# -*- coding: utf-8 -*-

from . import utils as u
import numpy as np
from symmetrize import sym, pointops as po
import scipy.ndimage as ndi
import poppy.zernike as ppz
import scipy.io as sio
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings as wn
try:
    from mpes import analysis as aly
except:
    wn.warn('The package mpes is not install, this could disable certain functionalities of the pacakge.')


def hexmask(hexdiag=128, imside=256, image=None, padded=False, margins=[], pad_top=None, pad_bottom=None,
            pad_left=None, pad_right=None, vertical=True, outside='nan', ret='mask', **kwargs):
    """ Generate a hexagonal mask. To use the function, either the argument ``imside`` or ``image`` should be
    given. The image padding on four sides could be specified with either ``margins`` altogether or separately
    with the individual arguments ``pad_xxx``. For the latter, at least two independent padding values are needed.

    **Parameters**\n
    hexdiag: int | 128
        Number of pixels along the hexagon's diagonal.
    imside: int | 256
        Number of pixels along the side of the (square) reference image.
    image: 2D array | None
        2D reference image to construct the mask for. If the reference (image) is given, each side
        of the generated mask is at least that of the smallest dimension of the reference.
    padded: bool | False
        Option to pad the image (need to set to True to enable the margins).
    margins: list/tuple | []
        Margins of the image [top, bottom, left, right]. Overrides the `pad_xxx` arguments.
    pad_top, pad_bottom, pad_left, pad_right : int, int, int, int | None, None, None, None
        Number of padded pixels on each of the four sides of the image.
    vertical: bool | True
        Option to align the diagonal of the hexagon with the vertical image axis.
    outside: numeric/str | 'nan'
        Pixel value outside the masked region.
    ret: str | 'mask'
        Return option ('mask', 'masked_image', 'all').
    """

    if image is not None:
        imshape = image.shape
        minside = min(imshape)
        mask = ppz.hexike_basis(nterms=1, npix=minside, vertical=vertical)[0,...]
    else:
        imshape = kwargs.pop('imshape', (imside, imside))
        mask = ppz.hexike_basis(nterms=1, npix=hexdiag, vertical=vertical)[0,...]

    # Use a padded version of the original mask
    if padded == True:

        # Padding image margins on all sides
        if len(margins) == 4:
            top, bottom, left, right = margins

        else:
            # Total padding pixel numbers along horizontal and vertical directions
            padsides = np.abs(np.asarray(imshape) - hexdiag)
            top, bottom = u.nonneg_sum_decomposition(a=pad_top, b=pad_bottom, absum=padsides[0])
            left, right = u.nonneg_sum_decomposition(a=pad_left, b=pad_right, absum=padsides[1])

        mask = np.pad(mask, ((top, bottom), (left, right)), mode='constant', constant_values=np.nan)

    if outside == 0:
        mask = np.nan_to_num(mask)

    if ret == 'mask':
        return mask
    elif ret == 'masked_image':
        return mask*image
    elif ret == 'all':
        margins = [top, bottom, left, right]
        return mask, margins


def decomposition_hex2d(band, bases=None, baxis=0, nterms=100, basis_type='Zernike', ret='coeffs'):
    """ Decompose energy band in 3D momentum space using the orthogonal polynomials in a hexagon.

    **Parameters**\n
    band: 2D array
        2D electronic band structure.
    bases: 3D array | None
        Matrix composed of bases to decompose into.
    baxis: int | 0
        Axis of the basis index.
    nterms: int | 100
        Number of basis terms.
    basis_type: str | 'Zernike'
        Type of basis to use.
    ret: str | 'coeffs'
        Options for the return values.
    """

    nbr, nbc = band.shape
    if nbr != nbc:
        raise ValueError('Input band surface should be square!')

    if bases is None:
        if basis_type == 'Zernike':
            bases = ppz.hexike_basis(nterms=nterms, npix=nbr, vertical=True, outside=0)
        elif basis_type == 'Fourier':
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        if baxis != 0:
            bases = np.moveaxis(bases, baxis, 0)

    nbas, nbasr, nbasc = bases.shape
    band_flat = band.reshape((band.size,))
    coeffs = np.linalg.pinv(bases.reshape((nbas, nbasr*nbasc))).T.dot(band_flat)

    if ret == 'coeffs':
        return coeffs
    elif ret == 'all':
        return coeffs, bases


def reconstruction_hex2d(coeffs, bases=None, baxis=0, npix=256, basis_type='Zernike', ret='band'):
    """ Reconstruction of energy band in 3D momentum space using orthogonal polynomials
    and the term-wise coefficients.

    **Parameters**\n
    coeffs: 1D array
        Polynomial coefficients to use in reconstruction.
    bases: 3D array | None
        Matrix composed of bases to decompose into.
    baxis: int | 0
        Axis of the basis index.
    npix: int | 256
        Number of pixels along one side in the square image.
    basis_type: str | 'Zernike'
        Type of basis to use.
    ret: str | 'band'
        Options for the return values.
    """

    coeffs = coeffs.ravel()
    nterms = coeffs.size

    if bases is None:
        if basis_type == 'Zernike':
            bases = ppz.hexike_basis(nterms=nterms, npix=npix, vertical=True, outside=0)
        elif basis_type == 'Fourier':
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        if baxis != 0:
            bases = np.moveaxis(bases, baxis, 0)

    nbas, nbasr, nbasc = bases.shape
    band_recon = bases.reshape((nbas, nbasr*nbasc)).T.dot(coeffs).reshape((nbasr, nbasc))

    if ret == 'band':
        return band_recon
    elif ret == 'all':
        return band_recon, bases


def projectionfilter(data, nterms=None, bases=None, npix=None, basis_type='Zernike',
                    outside='nan', basis_kwds={}):
    """ Filtering reconstructed band structure using orthogonal polynomial approximation.

    **Parameters**\n
    data: 2D array
        Band dispersion in 2D to filter.
    nterms: int | None
        Number of terms.
    bases: 3D array | None
        Bases for decomposition.
    npix: int | None
        Size (number of pixels) in one direction of each basis term.
    basis_type: str | 'Zernike'
        Type of basis to use for filtering.
    outside: numeric/str | 'nan'
        Values to fill for regions outside the Brillouin zone boundary.
    basis_kwds: dictionary | {}
        Keywords for basis generator (see `poppy.zernike.hexike_basis()` if hexagonal Zernike polynomials are used).
    """

    nterms = int(nterms)

    # Generate basis functions
    if bases is None:
        if basis_type == 'Zernike':
            bases = ppz.hexike_basis(nterms=nterms, npix=npix, **basis_kwds)

    # Decompose into the given basis
    coeffs = decomposition_hex2d(data, bases=bases, baxis=0,
                    nterms=nterms, basis_type=basis_type, ret='coeffs')

    # Reconstruct the smoothed version of the energy band
    recon = reconstruction_hex2d(coeffs, bases=bases, baxis=0,
                    npix=npix, basis_type=basis_type, ret='band')

    if outside == 'nan':
        recon = u.to_masked(recon, val=0)
        return recon, coeffs
    elif outside == 0:
        return recon, coeffs


def polydecompose(trace, deg, ids=None, method='piecewise', polytype='Legendre', **kwds):
    """ Decompose the trace into orthogonal polynomials.
    """

    nseg = len(ids)-1
    altids = u.segmod(ids)
    res = []

    if method == 'piecewise':

        for i in range(nseg):

            ida, idb = ids[i], altids[i+1]
            x = list(range(ida, idb))
            y = trace[ida:idb]

            if polytype == 'Legendre':
                res.append(np.polynomial.legendre.legfit(x, y, deg, **kwds))
            elif polytype == 'Chebyshev':
                res.append(np.polynomial.chebyshev.chebfit(x, y, deg, **kwds))

    elif method == 'complete':

        raise NotImplementedError

    try:
        res = np.asarray(res)
    except:
        pass

    return res


def polyreconstruct(coeffs, ids=None, polytype='Legendre', flatten=True):
    """ Reconstruct line segments using provided coefficients.
    """

    nseg = len(ids)-1
    res = []

    for i in range(nseg):

        ida, idb = ids[i], ids[i+1]
        x = list(range(ida, idb))
        cf = coeffs[i,:]

        if polytype == 'Legendre':
            res.append(np.polynomial.legendre.legval(x, cf))
        elif polytype == 'Chebyshev':
            res.append(np.polynomial.chebyshev.chebval(x, cf))

    if flatten == True:
        res = np.concatenate(res, axis=0)

    return res


def transdeform(imbase, xtrans=0, ytrans=0, interp_order=1, **kwargs):
    """ Image translation using deformation field.

    **Parameters**\n
    imbase: 2D array
        Base image before translation.
    xtrans, ytrans: numeric, numeric | 0, 0
        Magnitude of translation along the x and y axes.
    **kwargs: keyword arguments
        See additional arguments in `scipy.ndimage.map_coordinates()`.
    """

    coordmat = sym.coordinate_matrix_2D(imbase, coordtype='homogeneous', stackaxis=0)
    rdisp, cdisp = sym.translationDF(coordmat, stackaxis=0, ret='displacement', xtrans=xtrans, ytrans=ytrans)
    rdeform, cdeform = coordmat[1,...] + rdisp, coordmat[0,...] + cdisp
    imtrans = ndi.map_coordinates(imbase, [rdeform, cdeform], order=interp_order, **kwargs)

    return imtrans


def rotodeform(imbase, angle, center, interp_order=1, **kwargs):
    """ Image rotation using deformation field.

    **Parameters**\n
    imbase: 2D array
        Base image before rotation.
    angle: numeric
        Angle of rotation.
    center: list/tuple
        Center pixel coordinates of the image.
    **kwargs: keyword arguments
        See additional arguments in `scipy.ndimage.map_coordinates()`.

    **Return**\n
    imshift: 2D array
        Rotated image.
    """

    coordmat = sym.coordinate_matrix_2D(imbase, coordtype='homogeneous', stackaxis=0)
    rdisp, cdisp = sym.rotationDF(coordmat, stackaxis=0, ret='displacement', center=center, angle=angle)
    rdeform, cdeform = coordmat[1,...] + rdisp, coordmat[0,...] + cdisp
    imshift = ndi.map_coordinates(imbase, [rdeform, cdeform], order=interp_order, **kwargs)

    return imshift


def rotosymmetrize(image, center, rotsym=None, angles=None, outside='nan', **kwargs):
    """ Symmetrize the pattern according to the rotational symmetry.

    **Parameters**\n
    image: 2D array
        Image to symmetrize.
    center: list/tuple
        Image center pixel position (row, column).
    rotsym: int | None
        Order of rotation symmetry (if regular symmetry is assumed). If ``rotsym``
        is specified, the values from ``angles`` are ignored.
    angles: numeric | None
        Angles of rotation.
    outside: str/numeric | 'nan'
        The values of the symmetrized image outside the masked boundary.
    """

    image = np.nan_to_num(image)

    if rotsym is not None:
        rotsym = int(rotsym)
        angles = np.linspace(0, 360, rotsym, endpoint=False)

    # Generate symmetry equivalents
    rotoeqs = []
    for angle in angles:
        rotoeqs.append(rotodeform(imbase=image, angle=angle, center=center, **kwargs))
    rotoeqs = np.asarray(rotoeqs)
    rotoavg = rotoeqs.mean(axis=0)

    if outside == 'nan':
        rotoavg = u.to_masked(rotoavg, val=0)
        return rotoavg, angles
    elif outside == 0:
        return rotoavg, angles


def rotosymdetect(image, center, rotrange=list(range(-30, 330, 5)), lookahead=4,
                    pbar=True, pbenv='classic'):
    """ Detect the degree of rotational symmetry of an image.

    **Parameters**\n
    image: 2D array
        Image for rotational symmetry detection.
    center: list/tuple
        Image center coordinates.
    rotrange: list/tuple | list(range(-30, 330, 5))
        Rotation values to test.
    lookahead: int | 4
        Number of points ahead taken into consideration in peak detection.
    pbar: bool | True
        Option to show progress bar.
    pbenv: str | 'classic'
        Progress bar environment ('classic' or 'notebook').

    **Return**\n
    nmax: int
        Order of rotational symmetry.
    """

    val = []
    tqdm = u.tqdmenv(pbenv)

    for angle in tqdm(rotrange, disable=not(pbar)):
        imdf = rotodeform(image, angle, center=center)
        val.append(-np.linalg.norm(imdf - image))
    val = np.asarray(val)

    try:
        peaks = aly.peakdetect1d(val, x_axis=rotrange, lookahead=lookahead)
        nmax = len(peaks[0])
    except:
        nmax = 0

    return nmax


def hexfilter(images, center, axis=0, rotrange=list(range(-30, 330, 5)), lookahead=4,
                pbar=True, pbenv='classic', ret='all'):
    """ Filter out sixfold-symmetric images.

    **Parameters**\n
    images: 3D array
        Stack of 2D images.
    center: list/tuple/1D array
        Image center pixel coordinates.
    axis: int | 0
        Axis to extract images from stack.
    rotrange: list/tuple | list(range(-30, 330, 5))
        All rotations tested.
    lookahead: int | 4
        Number of points ahead taken into consideration in peak detection.
    pbar: bool | True
        Option to turn on/off progress bar.
    pbenv: str | 'classic'
        Notebook environment.
    ret: str | 'all'
        Option for return ('filtered' returns only filtered images, 'all' returns filtered images and the indices of symmetric images within the stack).
    """

    images = np.moveaxis(images, axis, 0)
    nimg = images.shape[0]
    symord = []
    tqdm = u.tqdmenv(pbenv)

    for i in tqdm(range(nimg), disable=not(pbar)):
        symord.append(rotosymdetect(images[i,...], center=center, rotrange=rotrange,
                        lookahead=lookahead, pbar=False))

    symord = np.asarray(symord)
    seq = np.where((symord > 5) & (symord <= 7))[0]

    hexbase = images[seq,...]

    if ret == 'filtered':
        return hexbase
    elif ret == 'all':
        return hexbase, symord, seq


def reflectodeform(imbase, refangle, center, axis=0, interp_order=1, **kwargs):
    """ Reflect the image with respect to the symmetry line across the image center
    using deformation field.

    **Parameters**\n
    imbase: 2D array
        Base image.
    refangle: numeric
        Reflection angle with respect to the image horizontal axis.
    center: list/tuple
        Center coordinates of the image.
    axis: int | 0
        Axis to reflect along.
    """

    imbase = np.nan_to_num(imbase)
    nr, nc = imbase.shape
    coordmat = sym.coordinate_matrix_2D(imbase, coordtype='homogeneous', stackaxis=0)

    R1 = sym.rotation2D(angle=refangle, center=center)
    if axis == 0:
        S = sym.scaling2D(xscale=1, yscale=-1)
        T = sym.translation2D(xtrans=0, ytrans=nr)
    elif axis == 1:
        S = sym.scaling2D(xscale=-1, yscale=1)
        T = sym.translation2D(xtrans=nc, ytrans=0)

    R2 = sym.rotation2D(angle=-refangle, center=center)
    M = np.linalg.multi_dot([R2, T, S, R1])

    rdeform, cdeform = sym.compose_deform_field(coordmat, M, stackaxis=0, ret='deformation', ret_indexing='rc')
    imshift = ndi.map_coordinates(imbase, [rdeform, cdeform], order=interp_order, **kwargs)

    return imshift


def reflectosymmetrize(image, center, refangles, axis=0, outside='nan'):
    """ Symmetrize the pattern according to reflection symmetry.
    """

    image = np.nan_to_num(image)

    # Generate reflection-equivalent images
    reflectoeqs = []
    for refangle in refangles:
        reflectoeqs.append(reflectodeform(imbase=image, refangle=refangle, center=center, axis=axis))
    reflectoeqs = np.asarray(reflectoeqs)
    reflectoavg = reflectoeqs.mean(axis=0)

    if outside == 'nan':
        reflectoavg = u.to_masked(reflectoavg, val=0)
        return reflectoavg
    elif outside == 0:
        return reflectoavg


def refsym(img, op='nanmax', op_package='numpy', axis=0, pbenv='classic', pbar=True):
    """ Symmetrize by reflections.
    """
    
    opr = u.pick_operator(op, package=op_package)
    tqdm = u.tqdmenv(pbenv)

    if axis != 0:
        imgsym = np.rollaxis(img, axis, 0)
    else:
        imgsym = img.copy()
    nimg = imgsym.shape[0]
    
    for i in tqdm(range(nimg), disable=not(pbar)):
        imcurr = imgsym[i,...]
        transviews = [imcurr, imcurr[::-1,:], imcurr[:,::-1], imcurr[::-1,::-1]]
        imgsym[i,...] = opr(np.asarray(transviews), axis=0)
    
    return imgsym


def cutedge(image, check_axis=1, boundary='square', ret='cutimage'):
    """ Cutting out the region beyond the edge of an image.

    **Parameters**\n
    image: 2D array
        Image (containing nan or 0 outside the region of interest) before cutting.
    check_axis: int | 1
        The long axis for determining the boundary.
    boundary: str | 'square'
        ``'square'`` Square image boundary.\n
        ``'tight'`` Tightest rectangular image boundary.
        The shape of the image boundary.
    ret: str | 'cutimage'
        Option to specify return quantity ('cutimage', 'cutrange', 'all').
    """

    image_alt = np.moveaxis(image, check_axis, 0)
    image_real = np.nan_to_num(image_alt)

    # Calculate the cut range along the row axis
    raxis_sum = image_real.sum(axis=0)
    indr_lower, indr_upper = u.nzbound(raxis_sum)
    edge = indr_upper - indr_lower
    half_edge = edge // 2

    # Calculate the cut range along the column axis
    caxis_sum = image_real.sum(axis=1)
    indc_lower, indc_upper = u.nzbound(caxis_sum)
    midpoint = (indc_upper + indc_lower) // 2

    # Cut edges of an image using the specified boundary condition.
    if boundary == 'square':
        indc_lower, indc_upper = midpoint - half_edge, midpoint + half_edge
        image_cut = image[indr_lower:indr_upper+1, indc_lower:indc_upper+1]
    elif boundary == 'tight':
        image_cut = image[indr_lower:indr_upper+1, indc_lower:indc_upper+1]

    cutrange = [indr_lower, indr_upper+1, indc_lower, indc_upper+1]
    if ret == 'cutimage':
        return image_cut
    elif ret == 'cutrange':
        return cutrange
    elif ret == 'all':
        return image_cut, cutrange


class BrillouinZoner(object):
    """ Class for truncating the band mapping data to the first Brillouin zone.
    """

    def __init__(self, folder='', bands=[], axis=0, mask=None, kvals=[[],[]]):

        self.folder = folder
        try:
            self.bands = np.moveaxis(bands, axis, 0)
        except:
            pass
        self.eaxis = axis
        self.mask = mask
        self.kvals = kvals

    def set_bands(self, bands):
        """ Set the energy bands.
        """

        self.bands = bands

    def set_kvals(self, kvals):
        """ Set the k values.
        """

        self.kvals = kvals

    def set_eaxis(self, axis):
        """ Set the index of the energy axis.
        """

        self.eaxis = axis

    def set_mask(self, mask):
        """ Set the mask for the energy band.
        """

        self.mask = mask

    def summary(self, rettype='dict'):
        """ A container of truncated band structure and parameters.

        **Parameters**\n
        rettype: str | 'dict'
            Data type of the returned summary (``'dict'`` or ``'list'``).
        """

        if rettype == 'dict':
            out = {'axes': {'kx':self.kvals[0], 'ky':self.kvals[1]}, 'bands': {'bands':self.bandcuts}}
            return out

        elif rettype == 'list':
            out = [['axes', {'kx':self.kvals[0], 'ky':self.kvals[1]}], ['bands', {'bands':self.bandcuts}]]
            return out

    @property
    def nbands(self):
        """ Number of bands.
        """

        try:
            nbs = self.bands.shape[0]
        except:
            nbs = 0

        return nbs

    def load_data(self, filename, loadfunc=None, ret=False, **kwargs):
        """ Load band structure data (energy values and momentum axes).
        """

        # Load the energy and momentum values of the electronic bands
        readout = loadfunc(self.folder + filename, **kwargs)
        self.bands = readout[0]
        self.kvals = readout[1:]

        if ret == True:
            return self.bands, self.kvals

    def select_slice(self, selector, axis=None):
        """ Select the image slice for landmark detection.

        **Parameters**\n
        selector: slice object
            A slice object for selection of image stacks for feature detection.
        """

        if axis is not None:
            self.set_eaxis(axis=axis)
            self.bands = np.moveaxis(self.bands, self.eaxis, 0)

        self.slice = self.bands[selector,...]
        if self.slice.ndim == 3:
            self.slice = self.slice.sum(axis=0)

    def set_landmarks(self, landmarks):
        """ Set the landmark locations for the image features.
        """

        self.landmarks = landmarks

    def findlandmarks(self, method='daofind', direction='ccw', center_det='centroidnn', ret=False, **kwargs):
        """ Determine the landmark locations, further details see ``mpes.analysis.peakdetect2d()``.

        **Parameters**\n
        method: str | 'daofind'
            Method for detecting landmarks ('daofind' or 'maxfind').
        direction: str | 'ccw'
            Direction to arrange the detected vertices ('cw' for clockwise or 'ccw' for counterclockwise).
        center_det: str | 'centroidnn'
            Method to determine the center position.
        ret: bool | False
            Option to return the outcome.
        **kwargs: keyword arguments
            image: 2D array | ``self.bands[0,...]``
                Image to extract landmark from.
            image_ofs: list/tuple | [0, 0, 0, 0]
                Systematic offsets applied to the detected landmarks.
        """

        img = kwargs.pop('image', self.bands[0,...])
        imofs = np.array(kwargs.pop('image_ofs', [0, 0, 0, 0]))
        img = u.cut_margins(img, imofs)

        self.landmarks = aly.peakdetect2d(img, method=method, **kwargs)
        self.landmarks += imofs[[0, 2]]

        if center_det is None:
            self.pouter = self.landmarks
            self.pcent = None
        else:
            self.pcent, self.pouter = po.pointset_center(self.landmarks, method=center_det, ret='cnc')
            self.pcent = tuple(self.pcent)
        # Order the point landmarks
        self.pouter_ord = po.pointset_order(self.pouter, direction=direction)

        if ret == True:
            return self.landmarks

    def maskgen(self, ret='all', **kwargs):
        """ Generate a mask using given parameters.

        **Parameters**\n
            See ``fuller.generator.hexmask()``.
        """

        imshape = kwargs.pop('imshape', self.slice.shape)
        self.mask, self.margins = hexmask(ret=ret, imshape=imshape, **kwargs)

    def resample(self, kvals, band, nx=None, ny=None, ret='all', **kwargs):
        """ Resample the energy band in a finer grid.

        **Parameters**\n
            See ``fuller.utils.interpolate2d()``.
        """

        rsband = u.interpolate2d(kvals[0][:,0], kvals[1][0,:], band,
                        nx=nx, ny=ny, ret=ret, **kwargs)

        return rsband

    def bandcutter(self, nx=None, ny=None, dmean=False, resampled=False, ret=False, **kwargs):
        """ Truncate the band within the first Brillouin zone.

        **Parameters**\n
        nx, ny: int, int | None, None
            Pixel numbers of the cut band along the image axes.
        dmean: bool | False
            Option to subtract the mean from the band structure.
        resampled: bool | False
            Option to resample the energy band in another k-grid.
        ret: bool | False
            Specifications for return values.
        **kwargs: keyword arguments
            mask: 2D array | ``self.mask``
                Mask matrix to apply to image.
            margins: list/tuple | ``self.margins``
                Four-sided margins for the truncated band structure.
            selector: list/slice object/None | None
                Selector along the band index axis.
            offsx, offsy: int, int | 0, 0
                Offsets to a square along x and y directions.
        """

        mask = kwargs.pop('mask', self.mask)
        margins = kwargs.pop('margins', self.margins)
        selector = kwargs.pop('selector', slice(0, self.nbands))
        offsx = kwargs.pop('offsx', 0)
        offsy = kwargs.pop('offsy', 0)
        bands = self.bands[selector,:,:]
        nbands = bands.shape[0]

        bandcuts = []
        for i in range(nbands):

            if resampled == True:
                # Augment the band structure
                band = self.resample(self.kvals, bands[i,...], nx=nx, ny=ny, ret='all', **kwargs)
            else:
                band = bands[i,...]
            # Construct the truncated band structure
            bandcut = u.cut_margins(band, margins, offsetx=offsx, offsety=offsy)
            bandcuts.append(bandcut)

        # Construct the truncated mask
        maskcut = u.cut_margins(mask, margins)
        # Mask out the band region outside the first Brillouin zone
        self.bandcuts = np.asarray(bandcuts) * maskcut[None,...]

        try: # likewise trim the extents of the k-values
            self.kvals[0] = self.kvals[0][margins[0], -margins[1]]
            self.kvals[1] = self.kvals[1][margins[2], -margins[3]]
        except:
            pass

        if dmean == True: # Subtract the mean value from band energies
            self.bandcuts -= np.nanmean(self.bandcuts, axis=(1,2))[:,None,None]

        if ret == 'cutbands':
            return self.bandcuts
        elif ret == 'all':
            return self.bandcuts, bands

    def save_data(self, form='h5', save_addr='./bandcuts.h5', **kwargs):
        """ Save truncated band structure data.

        **Parameters**\n
        form: str | 'h5'
            Format of the file to save.
        save_addr: str | './bandcuts'
            File-saving address.
        **kwargs: keyword arguments
            Additional arguments for the file-saving functions.
        """

        if form == 'mat':
            sio.savemat(save_addr, self.summary(rettype='dict'), **kwargs)

        elif form == 'h5':
            u.saveHDF(*self.summary(rettype='list'), save_addr=save_addr)

    def visualize(self, image, figsize=(4, 4), origin='lower',
                  annotated=False, points=None, scatterkws={}, **kwargs):
        """ Display (cut) bands.
        """

        f, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, origin=origin, **kwargs)

        # Add annotation to the figure
        if annotated:
            tsr, tsc = kwargs.pop('textshift', (3, 3))
            txtsize = kwargs.pop('textsize', 12)

            for pk, pvs in points.items():
                try:
                    ax.scatter(pvs[:,1], pvs[:,0], **scatterkws)
                except:
                    ax.scatter(pvs[1], pvs[0], **scatterkws)

                if pvs.size > 2:
                    for ipv, pv in enumerate(pvs):
                        ax.text(pv[1]+tsc, pv[0]+tsr, str(ipv), fontsize=txtsize)


class EBandSynthesizer(object):
    """ Class for synthesizing electronic band structure from basis functions.
    """

    def __init__(self, nbands, **kwargs):

        self.nbands = nbands
        self.bands = []
        self.kvals = [[], []]
        self.spacing = kwargs.pop('spacing', [])

        self.coeffs = kwargs.pop('coeffs', [])
        self.mask = kwargs.pop('mask', [])

    def set_kvals(self, kvals):
        """ Set momentum values.
        """

        self.kvals = kvals

    def set_mask(self, mask):
        """ Set the mask for the synthesized data.
        """

        self.mask = mask

    def set_nbands(self, nbands):
        """ Set the number of energy bands to synthesize.
        """

        self.nbands = nbands

    def set_spacing(self, spacing):
        """ Set the energy spacing between energy bands.
        """

        self.spacing = spacing

    def summary(self, rettype='dict'):
        """ A container of synthetic band structure and parameters.
        """

        if rettype == 'dict':
            out = {'axes':{'kx':self.kvals[0], 'ky':self.kvals[1]}, 'bands': {'band':self.bands}}
            return out

        elif rettype == 'list':
            out = [['axes', {'kx':self.kvals[0], 'ky':self.kvals[1]}], ['bands', {'band':self.bands}]]
            return out

    def basisgen(self, nterms, npix, vertical=True, outside=0, basis_type='Zernike'):
        """ Generate polynomial bases for energy band synthesis.
        """

        if basis_type == 'Zernike':
            self.bases = ppz.hexike_basis(nterms=nterms, npix=npix, vertical=vertical, outside=outside)

    def coeffgen(self, nterms, method='rand_gauss', **kwargs):
        """ Generate coefficients for energy band synthesis.
        """

        if method == 'rand_gauss':
            self.coeffs = np.random.randn(self.nbands, nterms, **kwargs)

    def synthesize(self, basis_type='Zernike', **kwargs):
        """ Generate 3D electronic band structure.
        """

        self.bands = []

        for n in range(self.nbands):
            coeffs = self.coeffs[n,...]
            self.bands.append(reconstruction_hex2d(coeffs, bases=self.bases, **kwargs))

        self.bands = np.asarray(self.bands)

    def save_bands(self, form, save_addr='', **kwargs):
        """ Save the synthesized energy bands.
        """

        if form == 'mat': # Save in mat format

            compression = kwargs.pop('mat_compression', False)
            sio.savemat(save_addr, self.summary(rettype='dict'), do_compression=compression, **kwargs)

        elif form in ('h5', 'hdf5'): # Save in hdf5 format

            u.saveHDF(*self.summary(rettype='list'), save_addr=save_addr)

    def visualize(self, selector=None, indaxis=0, backend='plotly', **kwargs):
        """ Plot synthesized band structure.
        """

        title = kwargs.pop('title', '')

        if backend == 'plotly':
            import bandana as bd

            fname = kwargs.pop('fname', '')
            bd.plotter.bandplot3d(self.bands, selector, indaxis=indaxis, title=title, fname=fname)

        elif backend == 'matplotlib':
            raise NotImplementedError


class MPESDataGenerator(object):
    """ Class for generating three-dimensional photoemssion data.
    """

    def __init__(self, bands, lineshape, baxis=0, **kwargs):

        if baxis != 0:
            bands = np.moveaxis(bands, baxis, 0)
        self.all_bands = bands
        self.bands = bands
        self.lineshape = lineshape

        try:
            self.nr, self.nc = self.bands[0,...].shape
        except:
            self.nr, self.nc = self.bands.shape

        self.amplitude = kwargs.pop('amplitude', [])
        self.sigma = kwargs.pop('srfwidth', [])
        self.gamma = kwargs.pop('linewidth', [])
        self.energy = kwargs.pop('energy', [])
        self.kvals = [[], []]
        self.data = []

    @property
    def parameters(self):
        """ A dictionary of lineshape parameters.
        """

        pars = {'amp':self.amplitude, 'xvar':self.energy, 'sig':self.sigma,
                'gam':self.gamma, 'ctr':self.bands}
        return pars

    @property
    def nbands(self):
        """ Number of bands used in the simulation.
        """

        bnd = self.bands.ndim

        if bnd == 2:
            return 1
        elif bnd > 2:
            return self.bands.shape[0]

    def summary(self, rettype='dict'):
        """ A container of synthetic band mapping data and parameters.
        """

        if rettype == 'dict':
            out = {'axes':{'E':self.energy, 'kx':self.kvals[0], 'ky':self.kvals[1]},
                    'binned':{'V':self.data}}
            return out

        elif rettype == 'list':
            out = [['axes', {'E':self.energy, 'kx':self.kvals[0], 'ky':self.kvals[1]}],
                    ['binned', {'V':self.data}]]
            return out

    def set_amplitude(self, amplitude):
        """ Set the amplitude of the lineshape function.
        """

        self.amplitude = amplitude

    def set_bands(self, bands):
        """ Set the energy band positions.
        """

        self.all_bands = bands

    def add_bands(self, bands, edir='lower'):
        """ Add an energy band the existing list.
        """

        if edir == 'lower':
            self.all_bands = np.concatenate((self.all_bands, bands))
        elif edir == 'higher':
            self.all_bands = np.concatenate((bands, self.all_bands))

    def select_bands(self, selector):
        """ Select energy bands by their indices.
        """

        self.bands = self.all_bands[selector,...]

    def set_matrix_elements(self, matelems):
        """ Set the matrix element intensity modulation in photoemission process.
        """

        self.matelems = matelems

    def set_kvals(self, kvals):
        """ Set the momentum values for the data.
        """

        self.kvals = kvals

    def set_lineshape(self, lineshape):
        """ Set the lineshape function.
        """

        self.lineshape = lineshape

    def set_energy(self, energy):
        """ Set the binding energy of the photoelectrons.
        """

        self.energy = energy

    def set_srfwidth(self, sigma):
        """ Set the width of the system response function (SRF).
        """

        self.sigma = sigma

    def set_linewidth(self, gamma):
        """ Set the intrinsic linewidth of electronic state.
        """

        self.gamma = gamma

    def generate_data(self, matrix_elements='off'):
        """ Generate photoemission data.
        """

        params = self.parameters.copy()
        params['xvar'] = self.energy[:,None,None]
        params['ctr'] = self.bands[:1,...]

        # Generate 3D data for at least 1 electronic band
        self.data = self.lineshape(feval=True, vardict=params)

        if self.nbands > 1:
            for b in range(1, self.nbands):
                params['ctr'] = self.bands[b,...]
                self.data += self.lineshape(feval=True, vardict=params)

        if matrix_elements == 'on':
            self.data = self.matelems[None,...]*self.data

    def save_data(self, form='h5', save_addr='', save_items='all', **kwargs):
        """ Save generated photoemission data.
        """

        dtyp = kwargs.pop('dtyp', 'float32')

        if form == 'mat': # Save as mat file (for Matlab)

            compression = kwargs.pop('mat_compression', False)
            sio.savemat(save_addr, self.summary(rettype='dict'), do_compression=compression, **kwargs)

        elif form in ('h5', 'hdf5'): # Save as hdf5 file

            u.saveHDF(*self.summary(rettype='list'), save_addr=save_addr)

        elif form == 'tiff': # Save as tiff stack

            try:
                import tifffile as ti
                ti.imsave(save_addr, data=self.data.astype(dtyp), **kwargs)
            except ImportError:
                raise ImportError('tifffile package is not installed locally!')

    def to_bandstructure(self):

        return


def hexpad(img, cvd, edgepad=None, **kwargs):
    """ Symmetrically pad an image in directions perpendicular to the hexagonal edges.
    
    **Parameters**\n
    img: 2d array
        Image to pad.
    cvd: numeric
        Center-vertex distance of the hexagon.
    edgepad: list/tuple
        Number of padded pixels on the edge of the image, ((left, right), (top, bottom)).
    **kwargs: keyword arguments
        op, op_package: str, str | 'nanmax', 'numpy'
            Namestring of the function and package using for image padding (package.function will be executed and applied when merging the original and the paddings).
        mask: str | 'numpy'
            Mask applied to the unpadded image before merging with the paddings (used to suppress the potential discontinuities of boundary pixels).
        edgevals: numeric | ``np.nan``
            Edge values outside the boundary of the mask.

        
    **Return**\n
    padded_view: 2d array
        Rectangular image after padding hexagonally.
    """
    
    op = kwargs.pop('op', 'nanmax')
    op_package = kwargs.pop('op_package', 'numpy')
    mask = kwargs.pop('mask', np.ones_like(img))
    edgevals = kwargs.pop('edgevals', np.nan)
    
    if edgepad is not None:
        img = np.pad(img, edgepad, mode='constant', constant_values=edgevals)
        mask = np.pad(mask, edgepad, mode='constant', constant_values=edgevals)
    
    ag = np.radians(30)
    cosa, sina = np.cos(ag), np.sin(ag)
    xt, yt = (cosa**2)*cvd, (cosa*sina)*cvd
    opr = u.pick_operator(op, package=op_package)

    # Translation and fill
    xyshifts = [(2*xt, -2*yt), (2*xt, 2*yt), (-2*xt, 2*yt), (-2*xt, -2*yt), (0, -4*yt), (0, 4*yt)]
    transviews = [transdeform((img*mask).T, xtrans=x, ytrans=y, cval=np.nan)
                    for (x, y) in xyshifts]
    padded_view = opr(np.asarray(transviews + [(img*mask).T]), axis=0)
    
    return padded_view.T


def hextiler(image, final_size, cvd, method='geometric', op='nanmax', op_package='numpy', ret='final'):
    """ Tiling the image plane with hexagonal patterns.

    **Parameters**\n
    image: 2D array
        Base image before hexagonal tiling.
    final_size: list/tuple
        Final size of the padded image (row_size, colum_size).
    cvd: numeric
        Center-vertex distance.
    method: str | 'geometric'
        Method for hexagonal tiling.
    op: str | 'nanmax'
        Namestring of the operator.
    op_package: str | 'numpy'
        Namestring of the software package to obtain the operator.
    ret: str | 'final'
        final: Return only the final result.
        all: Return results from all intermediate steps.
    """

    # Symmetric padding
    nr, nc = image.shape
    sympad = np.pad(image, ((nr-1, 0), (nc-1, 0)), mode='reflect', reflect_type='even')

    # Enlarge by padding nan values beyond the boundary
    spr, spc = sympad.shape
    fin_nr, fin_nc = final_size
    augr, augc = (fin_nr - spr) // 2, (fin_nc - spc) // 2
    impad = np.pad(sympad, ((augr, augr), (augc, augc)), mode='constant', constant_values=np.nan)

    if method == 'geometric':

        opr = u.pick_operator(op, package=op_package)
        nrp, ncp = impad.shape
        rp, cp = (nrp - 1) // 2, (ncp - 1) // 2

        # Rotation and fill
        impadrot = [rotodeform(impad, angle=i, center=(rp, cp), cval=np.nan) for i in [-60, 60]]
        rotviews = np.asarray(impadrot + [impad])
        rot_merged_view = opr(rotviews, axis=0)

        ag = np.radians(30)
        cosa, sina = np.cos(ag), np.sin(ag)
        xt, yt = (cosa**2)*cvd, (cosa*sina)*cvd

        # Translation and fill
        xyshifts = [(2*xt, -2*yt), (2*xt, 2*yt), (-2*xt, 2*yt), (-2*xt, -2*yt), (0, -4*yt), (0, 4*yt)]
        transviews = [transdeform(rot_merged_view, xtrans=x, ytrans=y, cval=np.nan)
                        for (x, y) in xyshifts]
        trans_merged_view = opr(np.asarray(transviews + [rot_merged_view]), axis=0)

    # Reiterate previous two steps (if needed)

    if ret == 'final':
        return trans_merged_view
    elif ret == 'all':
        return [trans_merged_view, rot_merged_view, impad, sympad]


def bandstack(data, baxis=2, nvb=None, ncb=None, gap_id=None, pbar=True, pbenv='classic', **kwargs):
    """ Construct a stack of energy bands after symmetrization.

    **Parameters**\n
    data: 3D array
        Patches of band structure data with the axes in any ordering of (kx, ky, band_index).
    baxis: int | 2
        Axis of the band index.
    nvb, ncb: int, int | None, None
        Number of valence and conduction bands to extract.
    gap_id: int | None
        Index number of the topmost valence band or bottommost conduction band,
        depending on the stacking order in the data variable.
    pbar: bool | True
        Option to turn on/off the progress bar in computation.
    pbenv: str | 'classic'
        Progress bar environment ('classic' or 'notebook').
    **kwargs: keyword arguments

    **Returns**\n
    vbands, cbands: 3D array, 3D array
        Stacked valence and conduction bands after symmetrization.
    """

    nvb, ncb, gap_id = int(nvb), int(ncb), int(gap_id)
    op = kwargs.pop('op', 'nanmax')
    tiler_ret = kwargs.pop('tiler_ret', 'final')
    final_size = kwargs.pop('final_size', [319, 339])
    cvd = kwargs.pop('cvd', 103.9)
    tqdm = u.tqdmenv(pbenv)

    data = np.moveaxis(data, baxis, 2)
    vbands, cbands = [], []

    if nvb is not None: # Process valence band data
        vbparts = data[..., :gap_id][...,::-1]

        for ivb in tqdm(range(nvb), disable=not(pbar)):
            vbands.append(hextiler(vbparts[...,ivb], final_size=final_size, cvd=cvd,
                            ret=tiler_ret, op=op, **kwargs))
        vbands = np.asarray(vbands)

    if ncb is not None: # Process conduction band data
        cbparts = data[..., gap_id:]

        for icb in tqdm(range(ncb), disable=not(pbar)):
            cbands.append(hextiler(cbparts[...,icb], final_size=final_size, cvd=cvd,
                            ret=tiler_ret, op=op, **kwargs))
        cbands = np.asarray(cbands)

    return vbands, cbands


def restore(img, **kwargs):
    """ Restore an image with irregularly distributed missing values (as nan's).

    **Parameters**\n
    img: nd array
        Multidimensional image array with missing data (as nan's).
    **kwargs: keyword arguments
        Additional arguments supplied to ``scipy.interpolate.griddata()``.
    """
    
    imgrst = img.copy()
    nanpos = np.where(np.isnan(img))
    realpos = np.where(np.invert(np.isnan(img)))
    
    interpval = interpolate.griddata(realpos, img[realpos], nanpos, **kwargs)
    imgrst[nanpos] = interpval
    
    return imgrst
