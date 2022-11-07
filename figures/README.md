# A machine learning route between band mapping and band structure

## Acronyms and formulas
- WSe$_2$ = tungsten diselenide
- 1BZ = first Brillouin zone
- DFT = density functional theory
- LDA, PBE, PBEsol, HSE06 = names of four different exchange-correlation functionals used for DFT calculations

## Installation and runtime
A docker image with all the dependencies can be build from **`/environment/Dockerfile`**. It takes around 4-6 min to build. Executing the **`run.sh`** for running the notebooks in **`/code`** (excluding the subfolders) takes 10-15 mins, depending on whether a GPU is available. Running the notebooks in the subfolders **`/code/extra`** and **`/code/recon`** requires first to uncomment the relevant notebooks in the **`run.sh`** file. Running the notebooks in **`/code/extra`** takes about 3 mins, while running those in **`/code/recon`** takes about 10-20 mins, depending on whether a GPU is available.


## Folder structure within the capsule
Information about the content within the folders of the compute capsule.

### Code folder
- **`/code/extra`** contains Jupyter notebooks for extra components of processing steps related to the current work.
- **`/code/recon`** contains Jupyter notebooks for demonstration of reconstruction.
- The rest of **`/code`** folder contains Jupyter notebooks illustrating the generation processes for the figures in the manuscript.
- The **`run.sh`** file executes all the notebooks directly under the **`/code`** folder but not those in its subfolders to reduce runtime.
- All notebooks can also be run interactively and independently through a Jupyter cloud workstation on the Code Ocean platform.

### Data folder
- **`/data/hyperparameter`** contains the tuned hyperparameters used for band structure reconstruction of WSe$_2$ photoemission data.
- **`/data/hyperparameter/tuning_SFig3`** contains hyperparameter tuning results used for Supplementary Figure 3.
- **`/data/pes`** contains photoemission band mapping data.
- **`/data/processed`** contains intermediately and partially processed photoemission data.
- **`/data/processed/hslines`** contains photoemission data sliced along high-symmetry lines of the WSe$_2$ Brillouin zone.
- **`/data/processed/patches`** contains the patches in the obtained by the reconstruction algorithm and line (pointwise) fitting of the energy distribution curves using the reconstruction outcome as initialization.
- **`/data/processed/wse2_recon`** contains the reconstructed and symmetrized bands of WSe$_2$.
- **`/data/processed/wse2_recon_1BZ`** contains the reconstructed and symmetrized bands of WSe$_2$ within the first Brillouin zone (hexagonal).
- **`/data/synthetic`** contains synthetic 3D multiband data generated using LDA-DFT calculation and associated reconstruction outcome.
- **`/data/synthetic/hse_lda`** contains per-band reconstruction outcome using HSE06-DFT calculations as initialization.
- **`/data/synthetic/pbe_lda`** contains per-band reconstruction outcome using PBE-DFT calculations as initialization.
- **`/data/synthetic/pbesol_lda`** contains per-band reconstruction outcome using PBEsol-DFT calculations as initialization.
- **`/data/synthetic/sc=0.8_lda`** contains per-band reconstruction outcome using 0.8$\times$ scaled LDA-DFT calculations as initialization.
- **`/data/synthetic/sc=1.2_lda`** contains per-band reconstruction outcome using 1.2$\times$ scaled LDA-DFT calculations as initialization.
- **`/data/theory`** contains DFT calculations of the band structure of WSe$_2$ at both original and processed stages.
- **`/data/theory/bands_1BZ`** contains the WSe$_2$ band structure within the first Brillouin zone (hexagonal).
- **`/data/theory/bands_padded`** contains the WSe$_2$ band structure of the first Brillouin zone and padded regions outside (square).
- **`/data/theory/hslines`** contains the WSe$_2$ band structure along high-symmetry lines.
- **`/data/theory/patch`** contains the WSe$_2$ band structure in the original patch calculated using DFT.

### Results folder
The following folders are generated when executing the **`run.sh`** script.
- **`/results/figures`** contains figures generated from the Jupyter notebooks under **`/code`** folder.
- **`/results/notebooks`** contains the executed Jupyter notebooks from the corresponding **`/code`** folder rendered in html format.
- **`/results/notebooks/extra`** contains the executed Jupyter notebooks from the corresponding **`/code/extra`** folder rendered in html format.
- **`/results/notebooks/recon`** contains the executed Jupyter notebooks from the corresponding **`/code/recon`** folder rendered in html format.