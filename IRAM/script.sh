#!/bin/bash

#cout << "./irlm <nKr> <nEv> <max-restarts> <diag> <tol> <spectrum: 0=LR, 1=SR> <threads> <res_type: 0=eig, 1=schur, 2=qr> <rot_type: 0=eig, 1=schur, 2=qr> <Sym_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>" << endl;

NKR=96
NEV=32
RESTARTS=100
DIAG=100
TOL=1e-10
SPECTRUM=0
THREADS=1
RES_TYPE=0
ROT_TYPE=0
SYM_TYPE=1
VERBOSITY=0
EIGEN_CHECK=0

./iram ${NKR} ${NEV} ${RESTARTS} ${DIAG} ${TOL} ${SPECTRUM} ${THREADS} ${RES_TYPE} ${ROT_TYPE} \
       ${SYM_TYPE} ${VERBOSITY} ${EIGEN_CHECK}

