# VOATAL

(Variations On A Theme Of Lanczos, "vote-ol") 11 Oct 2020

## Synopsis

VOATOL is a C++ playground for Lanczos and Arnoldi restarted eigensolvers.
Some OMP parallelism is employed, but the purpose of this repository
is to give the user an accessible entry point into how restarted
eigensolvers work. Ideally, the user would learn from this repository
and implement one of the solvers in their own framework.

## Build

VOATOL uses CMake. The standard CMake build instructions apply:

1. mkdir VOATOL_build
2. cd VOATOL_build
3. cmake ../VOATOL -DENABLE_OPENMP
4. make -j <N>

## Solvers

VOATOL offers three solvers, located in the 'test' directory. Each will construct
a random matrix and solve it using the given parameters. These are the parameters
common to all solvers:

mat_size <int> The rank of the square matrix to solve
nKr <int> The size of the Krylov space
nEv <int> The size of the compressed space
nConv <int> The number of converged eigenvalues to compute
max_restarts The maximum number of restarts
diag <double> Add this diagonal constant to the problem matrix
tol <double> Residual tolerance of the eigenvalues
spectrum <int> 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI
               with S=Smallest
	            L=Largest
		    M=Modulus
                    R=Real
	            I=Imaginary
verbosity <int> Give verbose output
                1=verbose, 0=quiet
Eigen Check <int> Cross check VOATOL's results against Eigen
                  0=false, 1=true




### Thick Restarted Lanczos Method (TRLM)

TRLM is used in the following fashion:

./trlm <mat_size> <nKr> <nEv> <nConv> <max-restarts> <diag> <tol> <amin> <amax>
       <polydeg> <spectrum: 0=LR, 1=SR> <LU> <batch>
       <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>

TRLM is a symmetric eigensolver, so we allow the use of Chebyshev polynomial
acceleration:

amin <double> The minimum of the polynomial
amax <double> The maximum of the polynomial
polydeg <int> The polynomial degree

One can learn more about chebyshev acceleration here:
https://github.com/lattice/quda/wiki/QUDA's-eigensolvers#polynomial-acceleration

We optionally use LU batched rotation of the Krylov space in TRLM. One
can rotate N vectors using only M extra vectors in memory. This is
useful in HPC applications where memory is limited.

LU <int> 0=false, 1=true 
batch <int> The batch size of the rotation

### Block Thick Restarted Lanczos Method (BLKTRLM)

This solver is the Block version of TRLM. It works by ensuring a block
of Krylov vectors remain orthonormal durung the algorithm. It is useful
in HPC applications where one can save time by moving the matrix data
to the registers once, while also moving multiple vectors. There is
no batched rotation option for BLKTRLM.

BLKTRLM is used in the following fashion:

./block_trlm <mat_size> <nKr> <nEv> <nConv> <max-restarts> <diag> <tol> <amin> <amax>
	     <polydeg> <spectrum: 0=LR, 1=SR> <block> <verbosity: 1=verbose, 0=quiet>
	     <Eigen Check: 0=false, 1=true>

The extra options are:

block <int> The size of the Kylov block to employ

### Implicilty Restarted Arnoldi Method (IRAM)

Arnoldi can solve hermitian and non-hermitian matrices. As such, we do not allow
polynomial acceleration to be employed, but one may solve for an hermitian
or non-hermitian matrix. The general usage is:

./irlm <mat_size> <nKr> <nEv> <nConv> <max-restarts> <diag> <tol>
       <spectrum: 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI> <mat_type: 0=asym, 1=sym>
       <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>

The extra options are:

mat_type <int> Use a symmetric or antisymmetric problem matrix
	       0=asym, 1=sym

Happy Hacking!