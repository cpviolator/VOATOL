#!/bin/bash

#./irlm <nKr> <nEv> <max-restarts> <diag> <tol> <spectrum: 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI> <threads> <qr_type: 0=arpack, 1=custom> <mat_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>

NKR=96
NEV=32
RESTARTS=100000
DIAG=100
TOL=1e-10
SPECTRUM=1
THREADS=1
ROT_TYPE=1
SYM_TYPE=0
VERBOSITY=0
EIGEN_CHECK=1

./iram ${NKR} ${NEV} ${RESTARTS} ${DIAG} ${TOL} ${SPECTRUM} ${THREADS} ${RES_TYPE} ${ROT_TYPE} \
       ${SYM_TYPE} ${VERBOSITY} ${EIGEN_CHECK}

