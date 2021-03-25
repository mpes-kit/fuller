#!/bin/bash

## Reconstruct 2 valence bands of WSe2
NITER=100 # Number of iteractions in reconstruction optimization
KSCALE=1
PYTHONPATH="/cygdrive/c/ProgramData/Anaconda3/python"
CODEPATH="./WSe2_K_recon.py"
FPATH="WSe2_K_recon.txt"
> $FPATH

BIDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14"
for BID in $BIDS
do
    echo "Tuning initial conditions for reconstructing band #$BID ..."
    echo "current band = $BID, energy shift = $SHFTS" >> $FPATH
    $PYTHONPATH $CODEPATH -bid=$BID -ksc=$KSCALE -pm='benchmark' -niter=$NITER -gpu=False >> $FPATH
    echo "" >> $FPATH
done