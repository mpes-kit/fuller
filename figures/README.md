# A machine learning route between band mapping and band structure

## Acronyms and formulas
- WSe$_2$ = tungsten diselenide
- 1BZ = first Brillouin zone
- DFT = density functional theory
- LDA, PBE, PBEsol, HSE06 = names of four different exchange-correlation functionals used for DFT calculations

## Folder structure
Information about the content within the folders of the compute capsule.

### Code folder
- **`figures/extra`** contains Jupyter notebooks for extra components of processing steps related to the current work.
- **`figures/recon`** contains Jupyter notebooks for demonstration of reconstruction.
- The rest of **`figures`** folder contains Jupyter notebooks illustrating the generation processes for the figures in the manuscript.

### Data folder
- **`data/hyperparameter`** contains the tuned hyperparameters used for band structure reconstruction of WSe$_2$ photoemission data.
- **`data/hyperparameter/tuning_SFig3`** contains hyperparameter tuning results used for Supplementary Figure 3.
- **`data/pes`** contains photoemission band mapping data.
- **`data/processed`** contains intermediately and partially processed photoemission data.
- **`data/processed/hslines`** contains photoemission data sliced along high-symmetry lines of the WSe$_2$ Brillouin zone.
- **`data/processed/patches`** contains the patches in the obtained by the reconstruction algorithm and line (pointwise) fitting of the energy distribution curves using the reconstruction outcome as initialization.
- **`data/processed/wse2_recon`** contains the reconstructed and symmetrized bands of WSe$_2$.
- **`data/processed/wse2_recon_1BZ`** contains the reconstructed and symmetrized bands of WSe$_2$ within the first Brillouin zone (hexagonal).
- **`data/synthetic`** contains synthetic 3D multiband data generated using LDA-DFT calculation and associated reconstruction outcome.
- **`data/synthetic/hse_lda`** contains per-band reconstruction outcome using HSE06-DFT calculations as initialization.
- **`data/synthetic/pbe_lda`** contains per-band reconstruction outcome using PBE-DFT calculations as initialization.
- **`data/synthetic/pbesol_lda`** contains per-band reconstruction outcome using PBEsol-DFT calculations as initialization.
- **`data/synthetic/sc=0.8_lda`** contains per-band reconstruction outcome using 0.8$\times$ scaled LDA-DFT calculations as initialization.
- **`data/synthetic/sc=1.2_lda`** contains per-band reconstruction outcome using 1.2$\times$ scaled LDA-DFT calculations as initialization.
- **`data/theory`** contains DFT calculations of the band structure of WSe$_2$ at both original and processed stages.
- **`data/theory/bands_1BZ`** contains the WSe$_2$ band structure within the first Brillouin zone (hexagonal).
- **`data/theory/bands_padded`** contains the WSe$_2$ band structure of the first Brillouin zone and padded regions outside (square).
- **`data/theory/hslines`** contains the WSe$_2$ band structure along high-symmetry lines.
- **`data/theory/patch`** contains the WSe$_2$ band structure in the original patch calculated using DFT.

### Results folder
- **`results/figures`** contains figures generated from the Jupyter notebooks under **`figures`** folder.
- **`results/notebooks`** contains the executed Jupyter notebooks from the corresponding **`figures`** folder rendered in html format.
- **`results/notebooks/extra`** contains the executed Jupyter notebooks from the corresponding **`figures/extra`** folder rendered in html format.
- **`results/notebooks/recon`** contains the executed Jupyter notebooks from the corresponding **`figures/recon`** folder rendered in html format.