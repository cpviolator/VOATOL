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

#define Nvec 1024
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
  if (argc < 12 || argc > 12) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./trlm <nKr> <nEv> <max-restarts> <diag> <tol> <amin> <amax> <polydeg> <spectrum: 0=LR, 1=SR> <LU> <batch>" << endl;
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
  bool LU = (atoi(argv[10]) == 1 ? true : false);
  int batch_size = atoi(argv[11]);
  
  if (!(nKr > nEv + 6)) {
    printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    exit(0);
  }

  printf("Mat size = %d\n", Nvec);
  printf("nKr = %d\n", nKr);
  printf("nEv = %d\n", nEv);
  printf("Restarts = %d\n", max_restarts);
  printf("diag = %e\n", diag);
  printf("tol = %e\n", tol);
  printf("reverse = %s\n", reverse == true ? "true" :  "false");
  
  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  MatrixXcd diagonal = MatrixXcd::Zero(Nvec, Nvec);
  diagonal *= diag;  
  
  // Copy to mat
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    for(int j=0; j<Nvec; j++) {
      mat[i][j] = ref(i,j);
      mat[i][j] += conj(ref(j,i));	
      if(i == j) mat[i][j] += diag;
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

  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  //Symmetric tridiagonal matrix
  std::vector<double> alpha(nKr, 0.0);
  std::vector<double>  beta(nKr, 0.0);

  //Residual vector. Also used as a temp vector
  std::vector<Complex*> r(1);
  r[0] = (Complex*)malloc(Nvec*sizeof(Complex));
  zero(r[0]);
  
  printf("START THICK RESTARTED LANCZOS SOLUTION\n");

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
  for(int i=0; i<Nvec; i++) r[0][i] = drand48();

  //Normalise initial source
  normalise(r[0]);
  
  //v_1
  copy(kSpace[0], r[0]);

  t1 = clock();

  // START TRLM
  // Thick restarted Lanczos method for symmetric eigenvalue problems
  //-----------------------------------------------------------------
  
  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {
    
    // (2) p = m-k steps to get to the m-step factorisation
    for (int step = num_keep; step < nKr; step++) {
      lanczosStep(mat, kSpace, beta, alpha, r, num_keep, step, a_min, a_max, poly_deg);
    }
    iter += (nKr - num_keep);
    
    printf("Restart %d complete\n", restart_iter+1);
    
    int arrow_pos = std::max(num_keep - num_locked + 1, 2);

    // The eigenvalues are returned in the alpha array
    eigensolveFromArrowMat(num_locked, arrow_pos, nKr, alpha, beta, residua, reverse);
    
    // mat_norm is updated.
    for (int i = num_locked; i < nKr; i++) {
      if (verbose) printf("fabs(alpha[%d]) = %e  :  mat norm = %e\n", i, fabs(alpha[i]), mat_norm);
      if (fabs(alpha[i]) > mat_norm) {
	mat_norm = fabs(alpha[i]);
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
    //iter_keep = std::min(max_keep, iter_keep);

    if(!LU) computeKeptRitz(kSpace, nKr, num_locked, iter_keep, beta);
    else computeKeptRitzLU(kSpace, nKr, num_locked, iter_keep, batch_size, beta, iter);
    
    num_converged = num_locked + iter_converged;
    num_keep = num_locked + iter_keep;
    num_locked += iter_locked;

    printf("iter Conv = %d\n", iter_converged);
    printf("iter Keep = %d\n", iter_keep);
    printf("iter Lock = %d\n", iter_locked);
    printf("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);
    printf("num_converged = %d\n", num_converged);
    printf("num_keep = %d\n", num_keep);
    printf("num_locked = %d\n", num_locked);
     
    // Check for convergence
    if (num_converged >= nEv) {
      reorder(kSpace, alpha, nKr, reverse);
      computeEvals(mat, kSpace, residua, evals, nKr);
      for(int i=0; i<nEv; i++) evals[i] -= diag;
      converged = true;
    }
    
    restart_iter++;
    
  }

  t2e = clock() - t1;
  
  // Post computation report  
  if (!converged) {    
    printf("TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space in %d "
	   "restart steps.\n",
	   nEv, nEv, nKr, max_restarts);
  } else {
    printf("TRLM computed the requested %d vectors in %d restart steps and %d OP*x operations in %e secs.\n", nEv,
	   restart_iter, iter, t2e/CLOCKS_PER_SEC);
    
    // Dump all Ritz values and residua
    for (int i = 0; i < nEv; i++) {
      printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, alpha[i], 0.0, residua[i]);
    }
    
    // Compute eigenvalues
    for (int i = 0; i < nKr; i++) alpha[i] = evals[i].real();
    //reorder(kSpace, alpha, nEv, reverse);
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(),
	     residua[i]);
    }

    for (int i = 0; i < nEv; i++) {
      //int idx = reverse ? (Nvec-1) - i : i;      
      int idx = i;
      //printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigensolverRef.eigenvalues()[idx])/eigensolverRef.eigenvalues()[idx]);
    }
  }
}
