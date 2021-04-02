#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import fuller
from fuller import mrfRec
import argparse, time
from tqdm import tqdm
import json
from hdfio import dict_io as io


# Definitions of command line interaction
parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('-bid', '--bindex', metavar='index', nargs='?', type=int, help='The index of band to reconstruct, needs an integer between 1 and 14')
parser.add_argument('-ksc', '--kscale', metavar='kscale', nargs='?', type=float, help='Momentum scaling')
parser.add_argument('-pm', '--parameters', metavar='parameters', nargs='?', type=str, help='(Hyper)Parameters used for reconstruction')
parser.add_argument('-pmfp', '--pmfpath', metavar='pmfpath', nargs='?', type=str, help='File path for user-defined (Hyper)Parameters')
parser.add_argument('-niter', '--numiter', metavar='numiter', nargs='?', type=int, help='Number of iterations in running the reconstruction')
parser.add_argument('-tc', '--timecount', metavar='timcount', nargs='?', type=bool, help='Whether to include time profiling')
parser.add_argument('-sft', '--eshift', metavar='eshift', nargs='*', help='Energy shift hyperparameters for initialization tuning')
parser.add_argument('-eta', '--eta', metavar='eta', nargs='*', help='Eta hyperparameters for model tuning')
parser.add_argument('-gpu', '--gpu', metavar='gpu', nargs='?', type=bool, help='Whether to use GPU for the optimization')
parser.set_defaults(bindex=1, kscale=1, parameters='benchmark', pmfpath='', numiter=100, timecount=True, eshift=[0], eta=[0], gpu=False)
cli_args = parser.parse_args()

# Band index
BID = cli_args.bindex
if BID < 1:
    BID = 1
# Momentum scaling
KSCALE = cli_args.kscale
# (Hyper)Parameters used for reconstruction ('benchmark', 'user', 'trial')
PARAMS = cli_args.parameters
# File path for user-defined (Hyper)Parameters
PMFPATH = cli_args.pmfpath
# Number of iterations in running the reconstruction
NITER = cli_args.numiter
# Option to include time profiling
TIMECOUNT = cli_args.timecount
# Energy shift hyperparameters used for initialization tuning
SHIFTS = list(map(float, cli_args.eshift))
# Eta (Gaussian width) hyperparameters used for model tuning
ETAS = list(map(float, cli_args.eta))
# Option to use GPU for the optimization
GPU = cli_args.gpu

if BID <= 2:
    # Bands 1-2
    erange = slice(10, 100)
elif (BID > 2) and (BID <= 4):
    # Bands 3-4
    erange = slice(10, 220)
elif (BID > 4) and (BID <= 8):
    # Bands 5-8
    erange = slice(10, 280)
elif (BID > 8) and (BID <= 14):
    # Bands 9-14
    erange = slice(10, 490)
else:
    erange = slice(10, 490)

# Load data and initialization
data_fname = r'../data/WSe2/synth/kpoint/kpoint_LDA_synth_14.h5'
data = io.h5_to_dict(data_fname)

theo_fname = r'../data/WSe2/theory/kpoint/kpoint_PBE.h5'
theo = io.h5_to_dict(theo_fname)
ky_theo, kx_theo = theo['kx'], theo['ky']
E_theo = theo['bands']

# Build MRF model
mrf = mrfRec.MrfRec(E=data['E'][erange], kx=data['kx'], ky=data['ky'], I=data['V'][..., erange], eta=0.03)
mrf.normalizeI(kernel_size=[5, 5, 20], n_bins=256, clip_limit=0.01)
mrf.I_normalized = True

# Load MRF parameters
if PARAMS == 'benchmark': # Benchmark hyperparameters in a json file
    with open(r'./Tuning_params_WSe2_K.json') as fparam:
        params = json.load(fparam)
    shifts = sorted(params['shifts'][str(BID)])
    etas = sorted(params['etas'][str(BID)])
elif PARAMS == 'user': # User-defined hyperparameters in a json file
    with open(PMFPATH) as fparam:
        params = json.load(fparam)
    shifts = sorted(params['shifts'][str(BID)])
    etas = sorted(params['etas'][str(BID)])
elif PARAMS == 'trial': # User-defined short list of hyperparameters directly from command line interface
    if len(SHIFTS) > 1:
        shifts = sorted(SHIFTS)
    else:
        shifts = [SHIFTS]
    
    if len(ETAS) > 1:
        etas = sorted(ETAS)
    else:
        etas = [ETAS]

reconmat = [] # Reconstruction outcome storage
dts = [] # Time difference

# Tuning hyperparameters for MRF reconstruction
for sh in tqdm(shifts):
    recon_part = []
    
    for eta in etas:
        mrf.eta = eta
        mrf.initializeBand(kx=kx_theo, ky=ky_theo, Eb=E_theo[BID-1,...], offset=sh, kScale=KSCALE, flipKAxes=True)
        
        # Estimate the time count
        t_start = time.time()
        mrf.iter_para(NITER, disable_tqdm=True, use_gpu=GPU)
        Erecon = mrf.getEb()
        t_end = time.time()
        dt = t_end - t_start
        dts.append(dt)
        
        recon_part.append(Erecon)
    reconmat.append(recon_part)

reconmat = np.array(reconmat)

if TIMECOUNT:
    dts = np.asarray(dts)
    dts_sum = np.sum(dts)
    print('Fitting took {} seconds'.format(dts_sum))

# Calculate the RMS error with respect to ground truth
rmsemat = np.linalg.norm(reconmat - data['data']['gt'][None,None,BID-1,...], axis=(2,3))

# Save results
np.savez(r'../results/WSe2_K_recon_{}_band_{}.npz'.format(PARAMS, str(BID).zfill(2)), reconmat=reconmat,
        rmsemat=rmsemat, shifts=shifts, etas=etas, params=params, kscale=KSCALE)