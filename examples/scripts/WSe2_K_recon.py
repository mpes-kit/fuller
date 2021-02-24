#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import fuller
from fuller import mrfRec
import argparse, time
from tqdm import notebook as nb
from hdfio import dict_io as io
import json


# Definitions of command line interaction
parser = argparse.ArgumentParser(description='Input arguments')
parser.add_argument('-bid', '--bindex', metavar='index', nargs='?', type=int, help='The index of band to reconstruct, needs an integer between 1 and 14')
parser.add_argument('-ksc', '--kscale', metavar='kscale', nargs='?', type=float, help='Momentum scaling')
parser.add_argument('-tc', '--timecount', metavar='timcount', nargs='?', type=bool, help='Whether to include time profiling')
parser.set_defaults(bindex=1, kscale=1, timecount=True)
cli_args = parser.parse_args()

# Band index
BID = cli_args.bindex
# Momentum scaling
KSCALE = cli_args.kscale
# Option to include time profiling
TIMECOUNT = cli_args.timecount

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

# Load data and initialization
data_fname = r'../data/WSe2/synth/kpoint_LDA_synth_14.h5'
data = io.h5_to_dict(data_fname)

theo_fname = r'../data/WSe2/theory/kpoint_PBE.h5'
theo = io.h5_to_dict(theo_fname)
ky_theo, kx_theo = theo['kx'], theo['ky']
E_theo = theo['bands']

# Build MRF model
mrf = mrfRec.MrfRec(E=data['E'][erange], kx=data['kx'], ky=data['ky'], I=data['V'][..., erange], eta=0.03)
mrf.normalizeI(kernel_size=[5, 5, 20], n_bins=256, clip_limit=0.01)
mrf.I_normalized = True

# Load MRF parameters
with open(r'./Tuning_params_WSe2_K.json') as fparam:
    params = json.load(fparam)
shifts = sorted(params['shifts'][str(BID)])
etas = sorted(params['etas'][str(BID)])

reconmat = []
t_start = time.time()

# Tuning hyperparameters for MRF reconstruction
for sh in nb.tqdm(shifts):
    recon_part = []
    
    for eta in etas:
        mrf.eta = eta
        mrf.initializeBand(kx=kx_theo, ky=ky_theo, Eb=E_theo[BID,...], offset=sh, kScale=KSCALE, flipKAxes=True)
        mrf.iter_para(100, disable_tqdm=True)
        Erecon = mrf.getEb()
        recon_part.append(Erecon)
    reconmat.append(recon_part)
reconmat = np.array(reconmat)

if TIMECOUNT:
    t_end = time.time()
    dt = t_end - t_start

# Calculate the RMS error with respect to ground truth
rmsemat = np.linalg.norm(reconmat - data['gt'][None,None,BID,...], axis=(2,3))