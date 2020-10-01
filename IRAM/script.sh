#!/bin/bash

#./irlm <nKr> <nEv> <max-restarts> <diag> <tol> <spectrum: 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI> <mat_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>

export OMP_NUM_THREADS=4

NKR=96
NEV=32
RESTARTS=100000
DIAG=100
TOL=1e-10
SPECTRUM=1
MAT_TYPE=2
VERBOSITY=0
EIGEN_CHECK=0

./iram ${NKR} ${NEV} ${RESTARTS} ${DIAG} ${TOL} ${SPECTRUM} ${MAT_TYPE} ${VERBOSITY} ${EIGEN_CHECK}

