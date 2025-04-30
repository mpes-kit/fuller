#! /usr/bin/env python
import contextlib
import warnings as wn

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import interpolate
from scipy import io
from scipy import ndimage
from tqdm import tqdm

from .generator import rotosymmetrize


class MrfRec:
    """Class for reconstructing band structure from band mapping data."""

    def __init__(
        self,
        E,
        kx=None,
        ky=None,
        I=None,
        E0=None,
        eta=0.1,
        includeCurv=False,
        etaCurv=0.1,
    ):
        """Initialize the class.

        **Parameters**\n
        E: 1D array | None
            Energy as numpy array.
        kx: 1D array | None
            Momentum along x axis as numpy array.
        ky: 1D array | None
            Momentum along y axis as numpy array.
        I: 3D array | None
            Measured intensity wrt momentum (rows) and energy (columns), generated if None.
        E0: numeric | None
            Initial guess for band structure energy values, if None mean of E is taken.
        eta: numeric | 0.1
            Standard deviation of neighbor interaction term
        includeCurv: bool | False
            Flag, if true curvature term is included during optimization.
        etaCurv: numeric | 0.1
            Standard deviation of curvature term.
        """

        # Check input
        if kx is None and ky is None:
            raise Exception("Either kx or ky need to be specified!")
        elif kx is None:
            kx = np.array([0.0])
        elif ky is None:
            ky = np.array([0.0])

        # Store data in object
        self.kx = kx.copy()
        self.ky = ky.copy()
        self.E = E.copy()
        self.lengthKx = kx.size
        self.lengthKy = ky.size
        self.lengthE = E.size
        self.I = I
        # Shift I because of log
        self.I -= np.min(self.I)
        self.I += np.min(self.I[self.I > 0])
        # Parameter for reconstruction
        self.eta = eta
        self.includeCurv = includeCurv
        self.etaCurv = etaCurv

        # Generate I if needed
        if I is None:
            self.generateI()

        # Initialize band structure
        if E0 is None:
            self.indEb = np.ones((self.lengthKx, self.lengthKx), int) * int(self.lengthE / 2)
        else:
            EE, EE0 = np.meshgrid(E, E0)
            ind1d = np.argmin(np.abs(EE - EE0), 1)
            self.indEb = ind1d.reshape(E0.shape)
        self.indE0 = self.indEb.copy()

        # Initialize change of log likelihood up to constant
        self.logP = np.array([self.getLogP()])
        self.epochsDone = 0

        # Set normalization flag
        self.I_normalized = False

    @classmethod
    def fromFile(cls, fileName, E0=None, eta=0.1):
        """Initialize reconstruction object from h5 file, returns reconstruction object initialized from h5 file.

        **Parameters**\n
        fileName: str
            Path to the file as string
        E0: numeric | None
            Initial guess for band structure energy values, if None mean of E is taken.
        """

        # Read data from file
        from mpes import fprocessing as fp

        data = fp.readBinnedhdf5(fileName)
        kx = data["kx"]
        ky = data["ky"]
        I = data["V"]
        if "E" in data:
            E = data["E"]
        else:
            tof = data["tof"]
            E = tof ** (-2)
            E -= np.min(E)
            E /= np.max(E)

        # Construct object
        return cls(E, kx, ky, I=I, E0=E0, eta=eta)

    @classmethod
    def loadBandsMat(cls, path):
        """Load bands from mat file in numpy matrix.

        **Parameters**\n
        path: str
            Path to the mat file.

        **Return**\n
            Tuple of momentum vectors and energy grid.
        """

        # Import data
        data = io.loadmat(path)

        # Save to numpy variables
        if np.abs(np.sum(np.diff(data["kxxsc"][:, 0]))) > np.abs(np.sum(np.diff(data["kxxsc"][0, :]))):
            kx = data["kxxsc"][:, 0]
            ky = data["kyysc"][0, :]
        else:
            kx = data["kxxsc"][0, :]
            ky = data["kyysc"][:, 0]
        evb = data["evb"]

        return (kx, ky, evb)

    def initializeBand(
        self,
        kx,
        ky,
        Eb,
        offset=0.0,
        flipKAxes=False,
        kScale=1.0,
        interp_method="linear",
    ):
        """Set E0 according to reference band, e.g. DFT calculation.

        **Parameters**\n
        kx, ky: 1D array, 1D array
            Momentum values for data along x and y directions.
        Eb: 1D array
            Energy values for band mapping data.
        offset: numeric | 0.
            Offset to be added to reference energy values.
        flipKAxes: bool | False
            Flag, if true the momentum axes of the references are interchanged.
        kxScale: numeric | 1.
            Scaling factor applied to k axes of reference band (after flipping if done).
        interp_method: str | 'linear'
            Method used to interpolate reference band on grid of measured data, 'linear' and 'nearest' are possible
            choices. Details see ``scipy.interpolate.RegularGridInterpolator()``.
        """

        # Detach data from input vars
        kx_in = kx.copy()
        ky_in = ky.copy()
        Eb_in = Eb.copy()

        # Preprocessing
        if flipKAxes:
            kx_in, ky_in = (ky_in, kx_in)
            Eb_in = np.transpose(Eb_in)

        # Scale axis
        self.kscale = kScale
        kx_in *= self.kscale
        ky_in *= self.kscale

        # Interpolation to grid of experimental data
        intFunc = interpolate.RegularGridInterpolator(
            (kx_in, ky_in),
            Eb_in,
            method=interp_method,
            bounds_error=False,
            fill_value=None,
        )
        kxx, kyy = np.meshgrid(self.kx, self.ky, indexing="ij")
        kxx = np.reshape(kxx, (self.lengthKx * self.lengthKy,))
        kyy = np.reshape(kyy, (self.lengthKx * self.lengthKy,))
        Einterp = intFunc(np.column_stack((kxx, kyy)))

        # Add shift to the energy values
        self.offset = offset
        self.E0 = np.reshape(Einterp + self.offset, (self.lengthKx, self.lengthKy))

        # Get indices of interpolated data
        EE, EE0 = np.meshgrid(self.E, self.E0)
        ind1d = np.argmin(np.abs(EE - EE0), 1)
        self.indEb = ind1d.reshape(self.E0.shape)
        self.indE0 = self.indEb.copy()

        # Reinitialize logP
        self.delHist()

    def smoothenI(self, sigma=(1.0, 1.0, 1.0)):
        """Apply a multidimensional Gaussian filter to the band mapping data (intensity values).

        **Parameters**\n
        sigma: list/tuple | (1, 1, 1)
            The vector containing the Gaussian filter standard deviations in pixel space for kx, ky, and E.
        """

        self.I = ndimage.gaussian_filter(self.I, sigma=sigma)

        # Reinitialize logP
        self.delHist()

    def normalizeI(
        self,
        kernel_size=None,
        n_bins=128,
        clip_limit=0.01,
        use_gpu=True,
        threshold=1e-6,
    ):
        """Normalizes the intensity using multidimensional CLAHE (MCLAHE).

        **Parameters**\n
        kernel_size: list/tuple | None
            Tuple of kernel sizes, 1/8 of dimension lengths of x if None.
        n_bins: int | 128
            Number of bins to be used in the histogram.
        clip_limit: numeric | 0.01
            Relative intensity limit to be ignored in the histogram equalization.
        use_gpu: bool | True
            Flag, if true gpu is used for computations if available.
        threshold: numeric | 1e-6
            Threshold below which intensity values are set to zero.
        """

        I_dtype = self.I.dtype

        try:
            from mclahe import mclahe

            self.I = mclahe(
                self.I,
                kernel_size=kernel_size,
                n_bins=n_bins,
                clip_limit=clip_limit,
                use_gpu=use_gpu,
            )
        except ImportError:
            wn.warn("The package mclahe is not installed, therefore no contrast enhancement is performed.")

        self.I = self.I / np.max(self.I)
        indSmall = self.I < threshold
        self.I[indSmall] = threshold
        self.I = self.I.astype(I_dtype)

        # Reinitialize logP
        self.delHist()

        # Update normalization flag
        self.I_normalized = True

    def symmetrizeI(self, mirror=True, rotational=True, rotational_order=6):
        """Symmetrize I with respect to reflection along x and y axis.

        **Parameters**\n
        mirror: bool | True
            Flag, if True mirror symmetrization is done wrt planes perpendicular to kx and ky axis.
        rotational: bool | True
            Flag, if True rotational symmetrization is done along axis at kx = ky = 0.
        rotational_order: int | 6
            Order of the rotational symmetry.
        """

        # Mirror symmetrization
        if mirror:
            # Symmetrize wrt plane perpendicular to kx axis
            indXRef = np.min(np.where(self.kx > 0.0)[0])
            lIndX = np.min([indXRef, self.lengthKx - indXRef])
            indX = np.arange(indXRef - lIndX, indXRef + lIndX)
            self.I[indX, :, :] = (self.I[indX, :, :] + self.I[np.flip(indX, axis=0), :, :]) / 2

            # Symmetrize wrt plane perpendicular to ky axis
            indYRef = np.min(np.where(self.ky > 0.0)[0])
            lIndY = np.min([indYRef, self.lengthKy - indYRef])
            indY = np.arange(indYRef - lIndY, indYRef + lIndY)
            self.I[:, indY, :] = (self.I[:, indY, :] + self.I[:, np.flip(indY, axis=0), :]) / 2

        # Rotational symmetrization
        if rotational:
            center = (np.argmin(np.abs(self.kx)), np.argmin(np.abs(self.ky)))
            for i in range(self.I.shape[2]):
                self.I[:, :, i], _ = rotosymmetrize(self.I[:, :, i], center, rotsym=rotational_order)

        # Reinitialize logP
        self.delHist()

    def generateI(self):
        pass

    def iter_seq(self, num_epoch=1, updateLogP=False, disable_tqdm=False):
        """Iterate band structure reconstruction process.

        **Parameters**\n
        num_epoch: int | 1
            Number of iterations.
        updateLogP: bool | False
            Flag, if true logP is updated every half epoch.
        disable_tqdm: bool | False
            Flag, it true no progress bar is shown during optimization.
        """

        # Prepare parameter for iteration
        logI = np.log(self.I)
        ENN = self.E / (np.sqrt(2) * self.eta)
        if self.includeCurv:
            ECurv = self.E / (np.sqrt(2) * self.etaCurv)

        # Do iterations
        indList = np.random.choice(self.lengthKx * self.lengthKy, self.lengthKx * self.lengthKy * num_epoch)
        for i, ind in enumerate(tqdm(indList, disable=disable_tqdm)):
            indx = ind // self.lengthKy
            indy = ind % self.lengthKy
            # Get logP for given index
            logP = 0
            if indx > 0:
                logP -= (ENN - ENN[self.indEb[indx - 1, indy]]) ** 2
                if self.includeCurv:
                    if indx > 1:
                        logP -= (ECurv[self.indEb[indx - 2, indy]] - 2 * ECurv[self.indEb[indx - 1, indy]] + ECurv) ** 2
                    if indx < (self.lengthKx - 1):
                        logP -= (ECurv[self.indEb[indx - 1, indy]] - 2 * ECurv + ECurv[self.indEb[indx + 1, indy]]) ** 2
            if indx < (self.lengthKx - 1):
                logP -= (ENN - ENN[self.indEb[indx + 1, indy]]) ** 2
                if self.includeCurv:
                    if indx < (self.lengthKx - 2):
                        logP -= (ECurv[self.indEb[indx - 2, indy]] - 2 * ECurv[self.indEb[indx - 1, indy]] + ECurv) ** 2
            if indy > 0:
                logP -= (ENN - ENN[self.indEb[indx, indy - 1]]) ** 2
                if self.includeCurv:
                    if indy > 1:
                        logP -= (ECurv[self.indEb[indx, indy - 2]] - 2 * ECurv[self.indEb[indx, indy - 1]] + ECurv) ** 2
                    if indy < (self.lengthKy - 1):
                        logP -= (ECurv[self.indEb[indx, indy - 1]] - 2 * ECurv + ECurv[self.indEb[indx, indy + 1]]) ** 2
            if indy < (self.lengthKy - 1):
                logP -= (ENN - ENN[self.indEb[indx, indy + 1]]) ** 2
                if self.includeCurv:
                    if indy < (self.lengthKy - 2):
                        logP -= (ECurv[self.indEb[indx, indy - 2]] - 2 * ECurv[self.indEb[indx, indy - 1]] + ECurv) ** 2
            logP += logI[indx, indy, :]
            self.indEb[indx, indy] = np.argmax(logP)
            if updateLogP and (
                ((i + 1) % (self.lengthKx * self.lengthKy)) == 0
                or ((i + 1) % (self.lengthKx * self.lengthKy)) == (self.lengthKx * self.lengthKy // 2)
            ):
                self.logP = np.append(self.logP, self.getLogP())

        self.epochsDone += num_epoch

    @tf.function
    def compute_logP(self, E1d, E3d, logI, indEb, lengthKx):
        squDiff = [[tf.square(tf.gather(E1d, indEb[i][j]) - E3d) for j in range(2)] for i in range(2)]
        logP = self._initSquMat(2)
        for i in range(2):
            for j in range(2):
                logP[i][j] = (
                    logI[i][j]
                    - squDiff[i - 1][j]
                    - squDiff[i][j - 1]
                    - tf.pad(
                        squDiff[i - 1][j][i : (lengthKx // 2 - 1 + i), :, :],
                        [[1 - i, i], [0, 0], [0, 0]],
                    )
                    - tf.pad(
                        squDiff[i][j - 1][:, j : (lengthKx // 2 - 1 + j), :],
                        [[0, 0], [1 - j, j], [0, 0]],
                    )
                )
        return logP

    @tf.function
    def compute_logPTot(self, logP, logI, indEb):
        return (
            tf.reduce_sum(tf.gather(logP[0][0], indEb[0][0], batch_dims=2))
            + tf.reduce_sum(tf.gather(logP[1][1], indEb[1][1], batch_dims=2))
            + tf.reduce_sum(tf.gather(logI[0][1], indEb[0][1], batch_dims=2))
            + tf.reduce_sum(tf.gather(logI[1][0], indEb[1][0], batch_dims=2))
        )

    @tf.function
    def compute_updateW(self, logP):
        # white Nodes
        updateW = [tf.argmax(logP[i][i], axis=2, output_type=tf.int32) for i in range(2)]

        return updateW

    @tf.function
    def compute_updateB(self, logP):
        # black Nodes
        updateB = [tf.argmax(logP[i][1 - i], axis=2, output_type=tf.int32) for i in range(2)]

        return updateB

    def iter_para(
        self,
        num_epoch=1,
        updateLogP=False,
        use_gpu=True,
        disable_tqdm=False,
    ):
        """Iterate band structure reconstruction process (no curvature), computations done in parallel using Tensorflow.

        **Parameters**\n
        num_epoch: int | 1
            Number of iteration epochs.
        updateLogP: bool | False
            Flag, if true logP is updated every half epoch
        use_gpu: bool | True
            Flag, if true gpu is used for computations if available
        disable_tqdm: bool | False
            Flag, it true no progress bar is shown during optimization
        """

        if use_gpu:
            physical_devices = tf.config.list_physical_devices("GPU")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        with contextlib.nullcontext() if use_gpu else tf.device("/CPU:0"):
            if updateLogP:
                self.logP = np.append(self.logP, np.zeros(2 * num_epoch))
            lengthKx = 2 * (self.lengthKx // 2)
            lengthKy = 2 * (self.lengthKy // 2)
            indX, indY = np.meshgrid(np.arange(lengthKx, step=2), np.arange(lengthKy, step=2), indexing="ij")
            logI = [[tf.constant(np.log(self.I[indX + i, indY + j, :])) for j in range(2)] for i in range(2)]
            indEb = [
                [tf.Variable(np.expand_dims(self.indEb[indX + i, indY + j], 2), dtype=tf.int32) for j in range(2)]
                for i in range(2)
            ]
            E1d = tf.constant(self.E / (np.sqrt(2) * self.eta))
            E3d = tf.constant(self.E / (np.sqrt(2) * self.eta), shape=(1, 1, self.E.shape[0]))

            logP = self.compute_logP(E1d, E3d, logI, indEb, lengthKx)

            for epoch in tqdm(range(num_epoch), disable=disable_tqdm):
                # white nodes
                updateW = self.compute_updateW(logP)
                for i in range(2):
                    indEb[i][i].assign(tf.expand_dims(updateW[i], 2))
                logP = self.compute_logP(E1d, E3d, logI, indEb, lengthKx)
                if updateLogP:
                    self.logP[2 * epoch + 1] = self.compute_logPTot(logP, logI, indEb).numpy()

                # black nodes
                updateB = self.compute_updateB(logP)
                for i in range(2):
                    indEb[i][1 - i].assign(tf.expand_dims(updateB[i], 2))
                logP = self.compute_logP(E1d, E3d, logI, indEb, lengthKx)
                if updateLogP:
                    self.logP[2 * epoch + 2] = self.compute_logPTot(logP, logI, indEb).numpy()

            # Extract results
            indEbOut = [[indEb_val.numpy()[:, :, 0] for indEb_val in indEb_row] for indEb_row in indEb]

        # Store results
        for i in range(2):
            for j in range(2):
                self.indEb[indX + i, indY + j] = indEbOut[i][j]

        self.epochsDone += num_epoch

    def iter_para_curv(
        self,
        num_epoch=1,
        updateLogP=False,
        use_gpu=True,
        disable_tqdm=False,
        graph_reset=True,
        **kwargs,
    ):
        """Iterate band structure reconstruction process (with curvature), computations done in parallel using
        Tensorflow.

        **Parameters**\n
        num_epoch: int | 1
            Number of iteration epochs.
        updateLogP: bool | False
            Flag, if true logP is updated every half epoch.
        use_gpu: bool | True
            Flag, if true gpu is used for computations if available.
        disable_tqdm: bool | False
            Flag, it true no progress bar is shown during optimization.
        """

        if not self.includeCurv:
            raise (Exception("Curvature is not considered in this MRF object. Please use iter_para instead."))

        # Preprocessing
        if updateLogP:
            self.logP = np.append(self.logP, np.zeros(3 * num_epoch))
        nKx = self.lengthKx // 3
        nKy = self.lengthKy // 3
        nE = self.lengthE
        lengthKx = 3 * nKx
        lengthKy = 3 * nKy
        indX, indY = np.meshgrid(np.arange(lengthKx, step=3), np.arange(lengthKy, step=3), indexing="ij")
        # Initialize logI and indEb for each field type
        logI = [[tf.constant(np.log(self.I[indX + i, indY + j, :])) for j in range(3)] for i in range(3)]
        indEb = [
            [tf.Variable(np.expand_dims(self.indEb[indX + i, indY + j], 2), dtype=tf.int32) for j in range(3)]
            for i in range(3)
        ]
        ENN1d = tf.constant(self.E / (np.sqrt(2) * self.eta))
        ENN3d = tf.constant(self.E / (np.sqrt(2) * self.eta), shape=(1, 1, self.E.shape[0]))
        ECurv1d = tf.constant(self.E / (np.sqrt(2) * self.etaCurv))
        ECurv3d = tf.constant(self.E / (np.sqrt(2) * self.etaCurv), shape=(1, 1, self.E.shape[0]))
        EbCurv = [[tf.gather(ECurv1d, indEb[i][j]) for j in range(3)] for i in range(3)]

        # Calculate square differences
        squDiff = [[tf.square(tf.gather(ENN1d, indEb[i][j]) - ENN3d) for j in range(3)] for i in range(3)]

        # Calculate log(P)
        logP = self._initSquMat(3)
        for i in range(3):
            for j in range(3):
                pi = [max(0, 1 - i), max(0, i - 1)]
                pj = [max(0, 1 - j), max(0, j - 1)]
                logP[i][j] = (
                    logI[i][j]
                    - tf.pad(
                        tf.slice(squDiff[i - 2][j], [pi[1], 0, 0], [nKx - pi[1], nKy, nE]),
                        [[0, pi[1]], [0, 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.slice(squDiff[i][j - 2], [0, pj[1], 0], [nKx, nKy - pj[1], nE]),
                        [[0, 0], [0, pj[1]], [0, 0]],
                    )
                    - tf.pad(
                        tf.slice(squDiff[i - 1][j], [0, 0, 0], [nKx - pi[0], nKy, nE]),
                        [[pi[0], 0], [0, 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.slice(squDiff[i][j - 1], [0, 0, 0], [nKx, nKy - pj[0], nE]),
                        [[0, 0], [pj[0], 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.square(
                            tf.slice(
                                tf.roll(EbCurv[i - 2][j], shift=1 - pi[1], axis=0)
                                - 2 * tf.roll(EbCurv[i - 1][j], shift=pi[0], axis=0)
                                + ECurv3d,
                                [1 - pi[1], 0, 0],
                                [nKx - 1 + pi[1], nKy, nE],
                            ),
                        ),
                        [[1 - pi[1], 0], [0, 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.square(
                            tf.slice(
                                tf.roll(EbCurv[i - 1][j], shift=pi[0], axis=0)
                                - 2 * ECurv3d
                                + tf.roll(EbCurv[i - 2][j], shift=-pi[1], axis=0),
                                [pi[0], 0, 0],
                                [nKx - pi[0] - pi[1], nKy, nE],
                            ),
                        ),
                        [[pi[0], pi[1]], [0, 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.square(
                            tf.slice(
                                ECurv3d
                                - 2 * tf.roll(EbCurv[i - 2][j], shift=-pi[1], axis=0)
                                + tf.roll(EbCurv[i - 1][j], shift=pi[0] - 1, axis=0),
                                [0, 0, 0],
                                [nKx - 1 + pi[0], nKy, nE],
                            ),
                        ),
                        [[0, 1 - pi[0]], [0, 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.square(
                            tf.slice(
                                tf.roll(EbCurv[i][j - 2], shift=1 - pj[1], axis=1)
                                - 2 * tf.roll(EbCurv[i][j - 1], shift=pj[0], axis=1)
                                + ECurv3d,
                                [0, 1 - pj[1], 0],
                                [nKx, nKy - 1 + pj[1], nE],
                            ),
                        ),
                        [[0, 0], [1 - pj[1], 0], [0, 0]],
                    )
                    - tf.pad(
                        tf.square(
                            tf.slice(
                                tf.roll(EbCurv[i][j - 1], shift=pj[0], axis=1)
                                - 2 * ECurv3d
                                + tf.roll(EbCurv[i][j - 2], shift=-pj[1], axis=1),
                                [0, pj[0], 0],
                                [nKx, nKy - pj[0] - pj[1], nE],
                            ),
                        ),
                        [[0, 0], [pj[0], pj[1]], [0, 0]],
                    )
                    - tf.pad(
                        tf.square(
                            tf.slice(
                                ECurv3d
                                - 2 * tf.roll(EbCurv[i][j - 2], shift=-pj[1], axis=1)
                                + tf.roll(EbCurv[i][j - 1], shift=pj[0] - 1, axis=1),
                                [0, 0, 0],
                                [nKx, nKy - 1 + pj[0], nE],
                            ),
                        ),
                        [[0, 0], [0, 1 - pj[0]], [0, 0]],
                    )
                )
        if updateLogP:
            logPTot = (
                tf.reduce_sum(tf.compat.v1.batch_gather(logP[0][0], indEb[0][0]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logP[1][1], indEb[1][1]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logP[2][2], indEb[2][2]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logI[0][1], indEb[0][1]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logI[1][0], indEb[1][0]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logI[2][0], indEb[2][0]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logI[0][2], indEb[0][2]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logI[2][1], indEb[2][1]))
                + tf.reduce_sum(tf.compat.v1.batch_gather(logI[1][2], indEb[1][2]))
            )

        # Do updates
        update = [
            [
                tf.compat.v1.assign(
                    indEb[i][j],
                    tf.expand_dims(tf.argmax(logP[i][j], axis=2, output_type=tf.int32), 2),
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
        updateW = [update[i][i] for i in range(3)]
        updateB = [update[(i + 1) % 3][i] for i in range(3)]
        updateO = [update[(i + 2) % 3][i] for i in range(3)]

        # Do optimization
        if use_gpu:
            config = kwargs.pop("config", None)
        else:
            config = kwargs.pop("config", tf.compat.v1.ConfigProto(device_count={"GPU": 0}))

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in tqdm(range(num_epoch), disable=disable_tqdm):
                sess.run(updateW)
                if updateLogP:
                    self.logP[3 * (i - num_epoch)] = sess.run(logPTot)
                sess.run(updateB)
                if updateLogP:
                    self.logP[3 * (i - num_epoch) + 1] = sess.run(logPTot)
                sess.run(updateO)
                if updateLogP:
                    self.logP[3 * (i - num_epoch) + 2] = sess.run(logPTot)

            # Extract results
            indEbOut = sess.run(indEb)

        # Store results
        for i in range(3):
            for j in range(3):
                self.indEb[indX + i, indY + j] = indEbOut[i][j][:, :, 0]

        self.epochsDone += num_epoch
        if graph_reset:
            tf.compat.v1.reset_default_graph()

    def getEb(self):
        """Retrieve the energy values of the reconstructed band."""

        return self.E[self.indEb].copy()

    def getLogP(self):
        """Retrieve the log likelihood of the electronic band structure given the model."""

        # Likelihood terms
        indKx, indKy = np.meshgrid(np.arange(self.lengthKx), np.arange(self.lengthKy), indexing="ij")
        logP = np.sum(np.log(self.I[indKx, indKy, self.indEb]))
        # Interaction terms
        Eb = self.getEb()
        if self.lengthKx > 1:
            logP -= np.sum((Eb[0 : (self.lengthKx - 1), :] - Eb[1 : self.lengthKx, :]) ** 2) / (2 * self.eta**2)
        if self.lengthKy > 1:
            logP -= np.sum((Eb[:, 0 : (self.lengthKy - 1)] - Eb[:, 1 : self.lengthKy]) ** 2) / (2 * self.eta**2)

        return logP

    def plotI(
        self,
        kx=None,
        ky=None,
        E=None,
        cmapName="viridis",
        plotBand=False,
        plotBandInit=False,
        bandColor="r",
        initColor="k",
        plotSliceInBand=False,
        figsize=[9, 9],
        equal_axes=False,
    ):
        """Plot the intensity against k and E.

        **Parameters**\n
        kx, ky: 1D array, 1D array | None, None
            kx, ky to plot respective slice.
        E: 1D array | None
            E to plot respective slice.
        plotBand: bool | False
            Flag, if true current electronic band is plotted in image.
        plotBandInit: bool | False
            Flag, if true E0 is plotted in image.
        bandColor: str | 'r'
            Color string for band for matplotlib.pyplot function.
        initColor: str | 'k'
            Color string for initial band for matplotlib.pyplot function.
        plotSliceInBand: bool | False
            Flag, if true plots band as colormesh and corresponding slice in red.
        figsize: list/tuple | [9, 9]
            size of the figure produced.
        equal_axes: bool | False
            use same scaling for both axes.
        """

        # Prepare data to plot
        if kx is not None:
            indKx = np.argmin(np.abs(self.kx - kx))
            x, y = np.meshgrid(self.ky, self.E)
            z = np.transpose(self.I[indKx, :, :])
            lab = [r"$k_y (\AA^{-1})$", "$E (eV)$"]
            Eb = self.getEb()
            E0 = self.E[self.indE0].copy()
            bandX = self.ky
            bandY = Eb[indKx, :]
            initY = E0[indKx, :]
        if ky is not None:
            indKy = np.argmin(np.abs(self.ky - ky))
            x, y = np.meshgrid(self.kx, self.E)
            z = np.transpose(self.I[:, indKy, :])
            lab = [r"$k_x (\AA^{-1})$", "$E (eV)$"]
            Eb = self.getEb()
            E0 = self.E[self.indE0].copy()
            bandX = self.kx
            bandY = Eb[:, indKy]
            initY = E0[:, indKy]
        if E is not None:
            indE = np.argmin(np.abs(self.E - E))
            x, y = np.meshgrid(self.kx, self.ky)
            z = np.transpose(self.I[:, :, indE])
            lab = [r"$k_x (\AA^{-1})$", r"$k_y (\AA^{-1})$"]

        # Plot I
        plt.rcParams["figure.figsize"] = figsize
        plt.figure()
        cmap = plt.get_cmap(cmapName)
        plt.pcolormesh(x, y, z, cmap=cmap)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(lab[0], fontsize=24)
        plt.ylabel(lab[1], fontsize=24)
        cb = plt.colorbar(pad=0.02)
        if self.I_normalized:
            colorbar_label = "$I/I_{max}$"
        else:
            colorbar_label = "$I (counts)$"
        cb.set_label(label=colorbar_label, fontsize=24)
        cb.ax.tick_params(labelsize=20)
        if equal_axes:
            ax = plt.gca()
            ax.set_aspect("equal", "box")

        # Plot band if requested
        if (plotBand or plotBandInit) and (E is None):
            if plotBand:
                plt.plot(bandX, bandY, bandColor, linewidth=2.0)
            if plotBandInit:
                plt.plot(bandX, initY, initColor, linewidth=2.0)
            if plotSliceInBand:
                x, y = np.meshgrid(self.kx, self.ky, indexing="ij")
                plt.figure()
                plt.pcolormesh(x, y, self.getEb())
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel(r"$k_x (\AA^{-1})$", fontsize=24)
                plt.ylabel(r"$k_y (\AA^{-1})$", fontsize=24)
                cb = plt.colorbar(pad=0.02)
                cb.ax.tick_params(labelsize=20)
                cb.set_label(label="$E (eV)$", fontsize=24)
                if equal_axes:
                    ax = plt.gca()
                    ax.set_aspect("equal", "box")
                if kx is not None:
                    plt.plot(
                        np.array(self.lengthKy * [self.kx[indKx]]),
                        self.ky,
                        "r",
                        linewidth=2.0,
                    )
                else:
                    plt.plot(
                        self.kx,
                        np.array(self.lengthKx * [self.ky[indKy]]),
                        "r",
                        linewidth=2.0,
                    )

    def plotBands(self, surfPlot=False, cmapName="viridis", figsize=[9, 9], equal_axes=False):
        """Plot reconstructed electronic band structure.

        **Parameters**\n
        surfPlot: bool | False
            Flag, if true a surface plot is shown in addition.
        cmapName: str | 'viridis'
            Name of the colormap.
        figsize: list/tuple | [9, 9]
            Size of the figure produced.
        equal_axes: bool | False
            Option to apply the same scaling for both axes.
        """

        x, y = np.meshgrid(self.kx, self.ky, indexing="ij")

        # Colormesh plot
        plt.rcParams["figure.figsize"] = figsize
        plt.figure()
        cmap = plt.get_cmap(cmapName)
        plt.pcolormesh(x, y, self.getEb(), cmap=cmap)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r"$k_x (\AA^{-1})$", fontsize=24)
        plt.ylabel(r"$k_y (\AA^{-1})$", fontsize=24)
        cb = plt.colorbar(pad=0.02)
        cb.ax.tick_params(labelsize=20)
        cb.set_label(label="$E (eV)$", fontsize=24)
        if equal_axes:
            ax = plt.gca()
            ax.set_aspect("equal", "box")

        # Surface plot
        if surfPlot:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.plot_surface(x, y, np.transpose(self.getEb()))
            ax.set_xlabel(r"$k_x (\AA^{-1})$", fontsize=24)
            ax.set_ylabel(r"$k_y (\AA^{-1})$", fontsize=24)
            ax.set_zlabel("$E (eV)$", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            for tick in ax.zaxis.get_major_ticks():
                tick.label.set_fontsize(20)

    def plotLoss(self):
        """Plot the change of the negative log likelihood."""

        epoch = np.linspace(0, self.epochsDone, len(self.logP), endpoint=True)
        plt.rcParams["figure.figsize"] = [9, 9]
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(epoch, -self.logP, linewidth=2.0)
        ax.set_xlabel("epochs", fontsize=24)
        ax.set_ylabel(r"$-\log(p)+$" + "const", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    def delHist(self):
        """Deletes the training history by resetting delta log(p) to its initial value."""

        self.logP = np.array([self.getLogP()])
        self.epochsDone = 0

    def saveBand(self, fileName, hyperparams=True, index=None):
        """Save the reconstructed electronic band and associated optimization parameters to file.

        **Parameters**\n
        fileName: str
            Name of the file to save data to.
        hyperparams: bool | True
            Option to save hyperparameters.
        index: int | None
            Energy band index.
        """

        with h5py.File(fileName, "w") as file:
            file.create_dataset("/axes/kx", data=self.kx)
            file.create_dataset("/axes/ky", data=self.ky)
            file.create_dataset("/bands/Einit", data=self.E0)
            file.create_dataset("/bands/Eb", data=self.getEb())

            if hyperparams:
                if index is None:
                    band_index = self.band_index
                else:
                    band_index = index
                file.create_dataset("/hyper/band_index", data=band_index)
                file.create_dataset("/hyper/k_scale", data=self.kscale)
                file.create_dataset("/hyper/E_offset", data=self.offset)
                file.create_dataset("/hyper/nn_eta", data=self.eta)

    def loadBand(self, Eb=None, fileName=None, use_as_init=True):
        """Load bands in reconstruction object, either using numpy matrix or directly from file.

        **Parameters**\n
        Eb: numpy array | None
            Energy values of an electronic band.
        fileName: str | None
            Name of h5 file containing band.
        use_as_init: bool | True
            Flag, if true loaded band is used as initialization of the object
        """

        if fileName is not None:
            file = h5py.File(fileName, "r")
            if self.lengthKx == file["/axes/kx"].shape[0] and self.lengthKy == file["/axes/ky"].shape[0]:
                Eb = np.asarray(file["/bands/Eb"])

        if Eb is not None:
            EE, EEb = np.meshgrid(self.E, Eb)
            ind1d = np.argmin(np.abs(EE - EEb), 1)
            self.indEb = ind1d.reshape(Eb.shape)

        # Set initial band
        if use_as_init:
            # Set indices of initial band
            self.indE0 = self.indEb.copy()

            # Reinitialize logP
            self.delHist()

    def _initSquMat(self, n, el=None):
        """Returns as square matrix of size nxn with el as element in each element.

        **Parameters**\n
        n: int
            Size of the square matrix.
        el: numeric | None
            Values of each element.

        **Return**\n
            Square matrix of size nxn with el as element in each element.
        """

        return [[el for _ in range(n)] for _ in range(n)]
