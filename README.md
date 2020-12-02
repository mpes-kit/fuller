# fuller
![License](https://img.shields.io/github/license/mpes-kit/fuller?color=lightgrey) ![PyPI Version](https://badge.fury.io/py/fuller.svg) ![Downloads](https://pepy.tech/badge/fuller)

Integrated computational framework for electronic band structure reconstruction and parametrization, powered by probabilistic machine learning

## Introduction

This Python package comprises a set of tools to reconstruct and parametrize the electronic band structure (EBS) from photoemission spectroscopy data. It implements the Markov Random Field model introduced in
[Xian & Stimper et al. (2020)](https://arxiv.org/abs/2005.10210) in TensorFlow.


## Methods of installation

The latest version of the package can be installed via pip

```
pip install --upgrade git+https://github.com/mpes-kit/fuller.git
```

Alternatively, download the repository and run

```
python setup.py install
```

Install directly from PyPI

```
pip install fuller
```

### Requirements

Apart from the packages specified in the `requirements.txt` file, `tensorflow` is needed. Installation instructions can be found at the [official webpage](https://www.tensorflow.org/install). The package works with the CPU only and GPU supported version of the framework. Currently, only version TensorFlow 1 (1.14 onwards) is supported, but we are working on porting it to TensorFlow 2.

## Sample dataset

As a model system to demonstrate the effectiveness of the methodology we worked on 3D photoemission data of the semiconductor tungsten diselenide (WSe<sub>2</sub>). It resolve the momentum along the x- and y- axis (k<sub>x</sub> and k<sub>y</sub>) and the energy.

### Reconstruction

All 14 valence band of WSe<sub>2</sub> are visible in the dataset. The optimization was initialized by DFT calculation with [HSE06](https://aip.scitation.org/doi/10.1063/1.1564060) hybrid exchange-correlation functional. The results are shown in the figure below.

![Valence bands of tungsten diselenide reconstructed using MRF model](https://github.com/VincentStimper/fuller/blob/master/images/mrf_rec_init_kx_slices.gif "Valence bands of tungsten diselenide reconstructed using MRF model")


## Documentation

Complete API documentation is provided [here](https://mpes-kit.github.io/fuller/).

### Preprocessing and Reconstruction

#### Class MrfRec

The `MrfRec` class is of central importance for reconstruction as well as preprocessing the data. To reconstruct the EBS create a `MrfRec` object and use its methods to perform the algorithms and plot the results. Here, we list a selection of the most important methods of the class. For further illustration on how to use the class check out the `mpes_reconstruction_mrf.ipynb` notebook in the example folder.

##### \_\_init\_\_

```python
def __init__(E, kx=None, ky=None, I=None, E0=None, eta=0.1, includeCurv=False, etaCurv=0.1):
    ...
```

* `E`: Energy as 1D numpy array
* `kx`: Momentum along x axis as 1D numpy array, if `None` it is set to 0
* `ky`: Momentum along y axis as 1D numpy array, if `None` it is set to 0
* `I`: Measured intensity wrt momentum (rows) and energy (columns), generated if `None`
* `E0`: Initial guess for band structure energy values, if `None` the median of `E` is taken
* `eta`: Standard deviation of neighbor interaction term
* `includeCurv`: Flag, if true curvature term is included during optimization
* `etaCurv`: Standard deviation of curvature term

##### iter_para

```python
def iter_para(num_epoch=1, updateLogP=False, use_gpu=True, disable_tqdm=False, graph_reset=False):
    ...
```

Hereby, the parallel optimization of Markov Random Field model can be performed to reconstruct an electronic
band.
* `num_epoch`: Number of epochs to perform
* `updateLogP`: Flag, if true logP is updated every half epoch (requires more computations)
* `use_gpu`: Flag, if true gpu is used for computations if available
* `disable_tqdm`: Flag, it true no progress bar is shown during optimization
* `graph_reset`: Flag, if true Tensorflow graph is reset after computation to reduce memory demand

##### normalizeI

```python
def normalizeI(kernel_size=None, n_bins=128, clip_limit=0.01, use_gpu=True, threshold=1e-6):
    ...
```

This performs Multidimensional Contrast Limited Adaptive Histogram Equalization (MCLAHE), introduced in [Stimper et al. 2019](https://ieeexplore.ieee.org/document/8895993). The method is a wrapper for the [TensorFlow implementation of the `mclahe` function](https://github.com/VincentStimper/mclahe).
* `kernel_size`: Tuple of kernel sizes, 1/8 of dimension lengths of x if `None`
* `n_bins`: Number of bins to be used in the histogram
* `clip_limit`: Relative intensity limit to be ignored in the histogram equalization
* `use_gpu`: Flag, if true gpu is used for computations if available
* `threshold`: Threshold below which intensity values are set to zero


## Citation

If you are using this package within your own projects, please cite it as
> R. P. Xian, V. Stimper, M. Zacharias, S. Dong, M. Dendzik, S. Beaulieu, B. Schölkopf, M. Wolf, L. Rettig, C. Carbogno, S. Bauer, and R. Ernstorfer, "A machine learning route between band mapping and band structure," arXiv:2005.10210, 2020.

Bibtex code
```
@article{Xian2020,
    author={R. P. Xian and V. Stimper and M. Zacharias and S. Dong and M. Dendzik and S. Beaulieu and
            B. Schölkopf and M. Wolf and L. Rettig and C. Carbogno and S. Bauer and R. Ernstorfer},
    journal={arXiv:2005.10210},
    title={A machine learning route between band mapping and band structure},
    year={2020},
}
```

