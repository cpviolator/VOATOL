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
#include <sys/time.h>

int mat_size = 128;

#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "algoHelpers.h"
#include "lapack.h"

int main(int argc, char **argv) {

  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);  
  // timing variables
  double t_total =  0.0;
  double t_init = 0;
  double t_sort = 0;
  double t_eigen = 0.0;
  double t_compute = 0.0;
  double t_EV = 0.0;  

  // START init
  //---------------------------------------------------------
  gettimeofday(&start, NULL);  
  if (argc < 15 || argc > 15) {
    cout << "./block_trlm <mat_size> <nKr> <nEv> <nConv> <max-restarts> <diag> <tol> <amin> <amax> <polydeg> <spectrum: 0=LR, 1=SR> <block> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>" << endl;
    exit(0);
  }

  mat_size = atoi(argv[1]);
  int nKr = atoi(argv[2]);
  int nEv = atoi(argv[3]);
  int nConv = atoi(argv[4]);
  int max_restarts = atoi(argv[5]);
  double diag = atof(argv[6]);
  double tol = atof(argv[7]);
  double a_min = atof(argv[8]);
  double a_max = atof(argv[9]);
  double poly_deg = atof(argv[10]);
  bool use_acc = (a_min == 0 || a_max == 0 || poly_deg == 0) ? false : true;
  int spectrum = atoi(argv[11]);
  bool reverse = spectrum == 1 ? true : false;
  int block_size = atoi(argv[12]);
  bool verbose = (atoi(argv[13]) == 1 ? true : false);
  bool eigen_check = (atoi(argv[14]) == 1 ? true : false);

  if (nKr != mat_size) {
    if (!(nKr > nEv + 6)) {
      printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
      exit(0);
    }
  } else if (nKr == mat_size) {
    printf("nKr=mat_size=%d Computing a complete Arnoldi factorisation\n", nKr);
  } else if (nEv > mat_size || nKr > mat_size) {
    printf("nKr=%d and nEv=%d must be less than mat_size=%d\n", nKr, nEv, mat_size);
    exit(0);
  }

  if (nKr%block_size != 0 || nEv%block_size != 0) {
    printf("block_size=%d must be a factor of both nKr=%d and nEv=%d\n", block_size, nKr, nEv);
    exit(0);
  }
  
  printf("Mat size = %d\n", mat_size);
  printf("nKr = %d\n", nKr);
  printf("nEv = %d\n", nEv);
  printf("block_size = %d\n", block_size);
  printf("Restarts = %d\n", max_restarts);
  printf("diag = %e\n", diag);
  printf("tol = %e\n", tol);
  printf("reverse = %s\n", reverse == true ? "true" :  "false");

  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(mat_size, mat_size);
  MatrixXcd diagonal = MatrixXcd::Identity(mat_size, mat_size);
  diagonal *= diag;  
  
  // Copy to mat
  Complex **mat = (Complex**)malloc(mat_size*sizeof(Complex*));
  for(int i=0; i<mat_size; i++) {
    mat[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    for(int j=0; j<mat_size; j++) {
      mat[i][j] = ref(i,j) + conj(ref(j,i));	
      if(i == j) mat[i][j] += diag;
    }
  }
  
  //Construct objects for Lanczos.
  //---------------------------------------------------------------------
  //Eigenvalues and their residuals
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);
  std::vector<double> arrow_eigs(nKr, 0.0);
  
  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr+block_size);
  for(int i=0; i<nKr+block_size; i++) {
    kSpace[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    zero(kSpace[i]);
  }

  //Symmetric block tridiagonal matrix
  std::vector<Complex> alpha(nKr * block_size, 0.0);
  std::vector<Complex>  beta(nKr * block_size, 0.0);
  
  //Residual vector. Also used as a temp vector
  std::vector<Complex*> r(block_size);
  for(int i=0; i<block_size; i++) {
    r[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    zero(r[i]);
  }
  
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
  // END init
  //---------------------------------------------------------
  gettimeofday(&end, NULL);  
  t_init += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------
  gettimeofday(&start, NULL);    
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigenSolver;
  std::vector<Complex> eigen_evals(mat_size, 0.0);
  if(eigen_check) {
    printf("START EIGEN SOLUTION\n");
    eigenSolver.compute(ref + ref.adjoint() + diagonal);  
    gettimeofday(&end, NULL);  
    t_eigen += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("END EIGEN SOLUTION\n");
    printf("Time to solve problem using Eigen = %e\n", t_eigen);
    for(int i=0; i<mat_size; i++) eigen_evals[i] = eigenSolver.eigenvalues()[i];
  }
  //-----------------------------------------------------------------------

  // START compute 
  //---------------------------------------------------------
  gettimeofday(&start, NULL);  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int b=0; b<block_size; b++) {
    for(int i=0; i<mat_size; i++) {
      r[b][i].real(drand48());
    }
  }
  
  //Gram-Schmidt initial sources
  gramSchmidt(r);
  orthoCheck(r, block_size, true);

  for(int b=0; b<block_size; b++) copy(kSpace[b], r[b]);
  t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

  // START BLKTRLM
  // Block Thick restarted Lanczos method for symmetric eigenvalue problems
  printf("START BLOCK THICK RESTARTED LANCZOS SOLUTION\n");
  //-----------------------------------------------------------------
  
  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {

    gettimeofday(&start, NULL);     
    for (int step = num_keep; step < nKr; step += block_size) {
      blockLanczosStep(mat, kSpace, beta, alpha, r, num_keep, step, block_size, a_min, a_max, poly_deg);
    }
    iter += (nKr - num_keep);
    gettimeofday(&end, NULL);  
    t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;    

    gettimeofday(&start, NULL);  
    int arrow_pos = num_keep - num_locked;
    // The eigenvalues are returned in the alpha_eigs array
    eigensolveFromBlockArrowMat(num_locked, arrow_pos, nKr, block_size, restart_iter, alpha, beta, arrow_eigs, residua, reverse);
    gettimeofday(&end, NULL);  
    t_EV += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    gettimeofday(&start, NULL);         
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
	if(verbose) printf("**** Locking %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], epsilon * mat_norm);
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
	if(verbose) printf("**** Converged %d resid=%+.6e condition=%.6e ****\n", i, residua[i + num_locked], tol * mat_norm);
	iter_converged = i;
      } else {
	// Unlikely to find new converged pairs
	break;
      }
    }

    iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - num_locked - 12);
    iter_keep = (iter_keep/block_size) * block_size;
    gettimeofday(&end, NULL);
    t_sort += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    gettimeofday(&start, NULL);    
    computeKeptRitzComplex(kSpace, nKr, num_locked, iter_keep, block_size, beta);
    gettimeofday(&end, NULL);  
    t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    
    num_converged = num_locked + iter_converged;
    num_converged = (num_converged/block_size) * block_size;
    
    num_keep = num_locked + iter_keep;
    num_keep = (num_keep/block_size) * block_size;
    
    num_locked += iter_locked;
    num_locked = (num_locked/block_size) * block_size;

    printf("%04d converged eigenvalues at restart iter %04d\n", num_converged, restart_iter + 1);
          
    if(verbose) {
      printf("iter Conv = %d\n", iter_converged);
      printf("iter Keep = %d\n", iter_keep);
      printf("iter Lock = %d\n", iter_locked);
      printf("num_converged = %d\n", num_converged);
      printf("num_keep = %d\n", num_keep);
      printf("num_locked = %d\n", num_locked);
    }
    // Check for convergence
    if (num_converged >= nConv) {
      gettimeofday(&start, NULL);
      for(int i=0; i<nKr; i++) evals[i].real(arrow_eigs[i]);
      reorder(kSpace, evals, residua, nKr, spectrum);
      computeEvals(mat, kSpace, residua, evals, nKr);
      gettimeofday(&end, NULL);  
      t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;      
      converged = true;
    }
    restart_iter++;    
  }
  
  gettimeofday(&total_end, NULL);  
  t_total = ((total_end.tv_sec  - total_start.tv_sec) * 1000000u + total_end.tv_usec - total_start.tv_usec) / 1.e6;
  
  // Post computation report  
  if (!converged) {    
    printf("BLOCK TRLM failed to compute the requested %d vectors with a %d search space and %d Krylov space and %d block size in %d restart steps and %d OPs"
	   "restart steps.\n",
	   nConv, nEv, nKr, block_size, max_restarts, iter);
  } else {
    printf("BLOCK TRLM computed the requested %d vectors with a %d search space and a %d Krylov space and %d block size in %d restart_steps and %d OPs in %e secs.\n", nConv, nEv, nKr, block_size, restart_iter, iter, (t_compute + t_sort + t_EV));
    
    // Dump all Ritz values and residua
    for (int i = 0; i < nConv && use_acc; i++) {
      printf("RitzValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, arrow_eigs[i], 0.0, residua[i]);
    }
    
    if(eigen_check) {
      // sort the eigen eigenvalues by the requested spectrum. The wanted values
      // will appear at the end of the array. We need a dummy residua array to use
      // the sorting function.
      std::vector<double> res_dummy(mat_size, 0.0);
      zsortc(spectrum+2, mat_size, eigen_evals, res_dummy);
      for (int i = 0; i < nConv; i++) {
	//int idx_e = mat_size - 1 - i;
	int idx_e = i;
	printf("EigenComp[%04d]: [(%+.8e, %+.8e) - (%+.8e, %+.8e)]/(%+.8e, %+.8e) = "
	       "(%+.8e,%+.8e)\n", i,
	       evals[i].real(), evals[i].imag(),
	       eigen_evals[idx_e].real(), eigen_evals[idx_e].imag(),
	       eigen_evals[idx_e].real(), eigen_evals[idx_e].imag(),
	       (evals[i].real() - eigen_evals[idx_e].real())/eigen_evals[idx_e].real(),
	       0.0);
      }
    } else {
      for (int i = 0; i < nConv; i++) {
	printf("EigValue[%04d]: ||(%+.8e, %+.8e)|| = %+.8e residual %.8e\n", i, evals[i].real(), evals[i].imag(), abs(evals[i]), residua[i]);
      }
    }
  }

  cout << "Timings:" << endl;
  if(eigen_check) cout << "Eigen = " << t_eigen << endl;
  cout << "init = " << t_init << endl;
  cout << "compute = " << t_compute << endl;
  cout << "sort = " << t_sort << endl;
  cout << "EV = " << t_EV << endl;
  cout << "missing = " << (t_total) << " - " << (t_compute + t_init + t_sort + t_EV + t_eigen) << " = " << (t_total - (t_compute + t_init + t_sort + t_EV + t_eigen)) << " ("<<(100*((t_total - (t_compute + t_init + t_sort + t_EV + t_eigen))))/t_total<<"%)" << endl;
  
}
