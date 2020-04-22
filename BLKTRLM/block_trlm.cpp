#include <vector>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <vector>
#include <cstring>
#include <cfloat>
#include <random>
#include <unistd.h>
#include <omp.h>

#define Nvec 256
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 11 || argc > 11) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./block_trlm <nKr> <nEv> <max-restarts> <diag> <tol> <amin> <amax> <polydeg> <spectrum: 0=LR, 1=SR> <block>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  double a_min = atof(argv[6]);
  double a_max = atof(argv[7]);
  double poly_deg = atof(argv[8]);
  bool reverse = (atoi(argv[9]) == 0 ? true : false);
  int block_size = atoi(argv[10]);
  
  if (!(nKr > nEv + 6)) {
    printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    exit(0);
  }

  if (nKr%block_size != 0 || nEv%block_size != 0) {
    printf("block_size=%d must be a factor of both nKr=%d and nEv=%d\n", block_size, nKr, nEv);
    exit(0);
  }
  
  printf("Mat size = %d\n", Nvec);
  printf("nKr = %d\n", nKr);
  printf("nEv = %d\n", nEv);
  printf("block_size = %d\n", block_size);
  printf("Restarts = %d\n", max_restarts);
  printf("diag = %e\n", diag);
  printf("tol = %e\n", tol);
  printf("reverse = %s\n", reverse == true ? "true" :  "false");
  
  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  MatrixXcd Diag = MatrixXcd::Zero(Nvec, Nvec);
  
  // Make ref Hermitian  
  for(int i=0; i<Nvec; i++) {
    Diag(i,i) = Complex(diag, 0.0);
  }
  
  // Copy to mat
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    ref(i,i) = Complex(diag, 0.0);
    mat[i][i] = ref(i,i);
    for(int j=0; j<i; j++) {
      mat[i][j] = ref(i,j);
      ref(j,i) = conj(ref(i,j));
      mat[j][i] = ref(j,i);
    }
  }
  
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigensolverRef(ref);
  cout << eigensolverRef.eigenvalues() << endl;
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //-----------------------------------------------------------------------

  //a_min = eigensolverRef.eigenvalues()[nKr+16];
  //a_max = eigensolverRef.eigenvalues()[Nvec-1]+1.0;

  //double a_min = 35;
  //double a_max = 150;
  
  //Construct objects for Lanczos.
  //---------------------------------------------------------------------
  //Eigenvalues and their residuals
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);
  std::vector<Complex> arrow_eigs(nKr, 0.0);
  
  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr+block_size);
  for(int i=0; i<nKr+block_size; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  //Symmetric block tridiagonal matrix
  std::vector<Complex> alpha(nKr * block_size, 0.0);
  std::vector<Complex>  beta(nKr * block_size, 0.0);

  
  //Residual vector. Also used as a temp vector
  std::vector<Complex*> r(block_size);
  for(int i=0; i<block_size; i++) {
    r[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(r[i]);
  }
  
  printf("START BLOCK TRLM SOLUTION\n");

  double epsilon = DBL_EPSILON;
  double mat_norm = 0.0;
  bool converged = false;
  int iter = 0;
  int restart_iter = 0;
  int iter_converged = 0;
  int iter_locked = 0;
  int iter_keep = 0;
  int num_converged = 0;
  int num_locked = 0;
  int num_keep = 0;

  // Populate source with randoms.
  printf("Using random guess\n");
  for(int b=0; b<block_size; b++) {
    for(int i=0; i<Nvec; i++) {
      r[b][i].real(drand48());
    }
  }
  
  //Gram-Schmidt initial sources
  gramSchmidt(r);
  orthoCheck(r, block_size, true);

  for(int b=0; b<block_size; b++) copy(kSpace[b], r[b]);

  t1 = clock();
  
  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {
    
    for (int step = num_keep; step < nKr; step += block_size) {
      blockLanczosStep(mat, kSpace, beta, alpha, r, num_keep, step, block_size, a_min, a_max, poly_deg);
    }
    iter += (nKr - num_keep);
    
    printf("Restart %d complete\n", restart_iter+1);

    int arrow_pos = num_keep - num_locked;

    // The eigenvalues are returned in the alpha array
    eigensolveFromBlockArrowMat(num_locked, arrow_pos, nKr, block_size, restart_iter, alpha, beta, arrow_eigs, residua, reverse);
    
    // mat_norm is updated.
    for (int i = num_locked; i < nKr; i++) {
      if (verbose) printf("abs(alpha[%d]) = %e  :  mat norm = %e\n", i, abs(alpha[i]), mat_norm);
      if (abs(alpha[i]) > mat_norm) {
	mat_norm = abs(alpha[i]);
      }
    }

    // Locking check
    iter_locked = 0;
    for (int i = 1; i < (nKr - num_locked); i++) {
      if (residua[i + num_locked] < epsilon * mat_norm) {
	printf("**** Locking %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], epsilon * mat_norm);
	iter_locked = i;
      } else {
	// Unlikely to find new locked pairs
	break;
      }
    }

    // Convergence check
    iter_converged = iter_locked;
    for (int i = iter_locked + 1; i < nKr - num_locked; i++) {
      if (residua[i + num_locked] < tol * mat_norm) {
	printf("**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], tol * mat_norm);
	iter_converged = i;
      } else {
	// Unlikely to find new converged pairs
	break;
      }
    }

    iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - num_locked - 12);
    iter_keep = (iter_keep/block_size) * block_size;

    computeKeptRitzComplex(kSpace, nKr, num_locked, iter_keep, block_size, beta);
    
    num_converged = num_locked + iter_converged;
    num_converged = (num_converged/block_size) * block_size;
    
    num_keep = num_locked + iter_keep;
    num_keep = (num_keep/block_size) * block_size;
    
    num_locked += iter_locked;
    num_locked = (num_locked/block_size) * block_size;
    
    printf("iter Conv = %d\n", iter_converged);
    printf("iter Keep = %d\n", iter_keep);
    printf("iter Lock = %d\n", iter_locked);
    printf("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);
    printf("num_converged = %d\n", num_converged);
    printf("num_keep = %d\n", num_keep);
    printf("num_locked = %d\n", num_locked);
     
    // Check for convergence
    if (num_converged >= nEv) converged = true;
    
    restart_iter++;    
  }

  t2e = clock() - t1;
  
  // Post computation report  
  if (!converged) {    
    printf("BLOCK TRLM failed to compute the requested %d vectors with a %d search space, %d block size, and %d Krylov space in %d "
	   "restart steps.\n",
	   nEv, nEv, block_size, nKr, max_restarts);
  } else {
    printf("BLOCK TRLM computed the requested %d vectors with %d block size in %d restart steps and %d OP*x operations in %e secs.\n", nEv, block_size, restart_iter, iter, t2e/CLOCKS_PER_SEC);
    
    // Dump all Ritz values and residua
    for (int i = 0; i < nEv; i++) {
      printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, arrow_eigs[i].real(), arrow_eigs[i].imag(), residua[i]);
    }
    
    // Compute eigenvalues
    computeEvals(mat, kSpace, residua, evals, nKr);
    for (int i = 0; i < nKr; i++) alpha[i] = evals[i].real();
    //reorder(kSpace, alpha, nEv, reverse);
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(),
	     residua[i]);
    }

    for (int i = 0; i < nEv; i++) {
      int idx = i;
      printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigensolverRef.eigenvalues()[idx])/eigensolverRef.eigenvalues()[idx]);
    }
  }
}
