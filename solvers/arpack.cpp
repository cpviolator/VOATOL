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
#include <arpack_interface.h>

int mat_size = 128;

#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
//Eigen::IOFormat CleanFmt(16, 0, ", ", "\n", "[", "]");

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "lapack.h"
#include "algoHelpers.h"
#include "io.h"

int main(int argc, char **argv) {

  struct timeval start, end, total_start, total_end;
  gettimeofday(&total_start, NULL);  
  // timing variables
  double t_total =  0.0;
  double t_init = 0;
  double t_sort = 0;
  double t_eigen = 0.0;
  double t_compute = 0.0;
  double t_ARPACK = 0.0;
  double t_IO = 0.0;  

  // START init
  //---------------------------------------------------------
  if (argc < 12 || argc > 12) {
    cout << "./arpack <mat_size> <nKr> <nEv> <nConv> <max-restarts> <diag> <tol> <spectrum: 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI> <mat_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>" << endl;
    exit(0);
  }
  mat_size = atoi(argv[1]);
  int nKr = atoi(argv[2]);
  int nEv = atoi(argv[3]);
  int nConv = atoi(argv[4]);
  int max_restarts = atoi(argv[5]);
  double diag = atof(argv[6]);
  double tol = atof(argv[7]);
  int spectrum = atoi(argv[8]);
  bool symm = (atoi(argv[9]) == 1 ? true : false);
  bool verbose = (atoi(argv[10]) == 1 ? true : false);
  bool eigen_check = (atoi(argv[11]) == 1 ? true : false);

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

  printf("Mat size = %d\n", mat_size);
  printf("nKr = %d\n", nKr);
  printf("nEv = %d\n", nEv);
  printf("nConv = %d\n", nConv);
  printf("Restarts = %d\n", max_restarts);
  printf("diag = %e\n", diag);
  printf("tol = %e\n", tol);
    
  // Construct a matrix using Eigen.
  //---------------------------------------------------------------------
  MatrixXcd ref = MatrixXcd::Random(mat_size, mat_size);
  MatrixXcd diagonal = MatrixXcd::Identity(mat_size, mat_size);
  diagonal *= diag;
  
  //Copy Eigen ref matrix.
  Complex **mat = (Complex**)malloc(mat_size*sizeof(Complex*));
  for(int i=0; i<mat_size; i++) {
    mat[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    for(int j=0; j<mat_size; j++) {
      mat[i][j] = ref(i,j);
      if(symm) mat[i][j] += conj(ref(j,i));
      if(i == j) mat[i][j] += diag;
    }
  }

  //Construct objects for ARPACK
  //------------------------------------------  
  // all FORTRAN communication uses underscored 
  int ido_;
  int info_;
  int *ipntr_ = (int*)malloc(11*sizeof(int));
  int *iparam_ = (int*)malloc(11*sizeof(int));
  int n_    = mat_size,
    nev_    = nEv,
    nkv_    = nKr,
    ldv_    = mat_size,
    lworkl_ = (3 * nKr * nKr + 5 * nKr) * 2,
    rvec_   = 1;
  int max_iter = max_restarts;

  double tol_ = tol;

  //ARPACK workspace
  Complex sigma_ = 0.0;
  Complex *resid_ = (Complex *) malloc(ldv_*sizeof(Complex));
  Complex *w_workd_ = (Complex *) malloc(3*ldv_*sizeof(Complex));
  Complex *w_workl_ = (Complex *) malloc(lworkl_*sizeof(Complex)); 
  Complex *w_workev_= (Complex *) malloc(3*nkv_*sizeof(Complex));
  double *w_rwork_ = (double *) malloc(nkv_*sizeof(double));    
  int *select_ = (int*)malloc(nkv_*sizeof(int));
  
  Complex *evecs = (Complex *) malloc(nkv_*n_*sizeof(Complex));
  Complex *evals = (Complex *) malloc(nkv_   *sizeof(Complex));

  Complex one(1.0,0.0);
  
  for(int n=0; n<nkv_; n++) {
    evals[n] = 0;
    for(int i=0; i<n_; i++) {
      evecs[n*n_ + i] = 0;
      if(n==0) {
	resid_[i].real(drand48());
	resid_[i].imag(drand48());
      }
    }
  }
  
  //Alias pointers
  Complex *evecs_ = nullptr;
  evecs_ = (Complex*)(evecs);    
  Complex *evals_ = nullptr;
  evals_ = (Complex*)(evals);
  
  //Memory checks
  if((iparam_ == nullptr) ||
     (ipntr_ == nullptr) || 
     (resid_ == nullptr) ||  
     (w_workd_ == nullptr) || 
     (w_workl_ == nullptr) ||
     (w_workev_ == nullptr) ||
     (w_rwork_ == nullptr) || 
     (select_ == nullptr) ) {
    printf("eigenSolver: not enough memory for ARPACK workspace.\n");
    exit(0);
  }    

  //Assign values to ARPACK params 
  ido_        = 0;
  info_       = 1;
  iparam_[0]  = 1;
  iparam_[2]  = max_iter;
  iparam_[3]  = 1;
  iparam_[6]  = 1;
  
  //ARPACK problem type to be solved
  char howmany = 'A';
  char bmat = 'I';
  char spectrum_char[3] = {'S','M'};
  int iter_cnt= 0;
 
  Complex *psi1;
  Complex *psi2;

  Complex *psi1_cpy = (Complex*)malloc(n_*sizeof(Complex));
  Complex *psi2_cpy = (Complex*)malloc(n_*sizeof(Complex));
  
  for(int i=0; i<n_; i++) {
    psi1_cpy[i] = 1.0;
    psi2_cpy[i] = 1.0;
  }
  
  psi1 = w_workd_;
  psi2 = w_workd_ + n_;
  // END init
  //---------------------------------------------------------
  gettimeofday(&end, NULL);  
  t_init += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

  // Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------
  gettimeofday(&start, NULL);  
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolver;
  std::vector<Complex> eigen_evals(mat_size, 0.0);
  if(eigen_check) {
    printf("START EIGEN SOLUTION\n");
    if(symm) eigenSolver.compute(ref + ref.adjoint() + diagonal);
    else eigenSolver.compute(ref + diagonal);
    gettimeofday(&end, NULL);  
    t_eigen += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    printf("END EIGEN SOLUTION\n");
    printf("Time to solve problem using Eigen = %e\n", t_eigen);
    for(int i=0; i<mat_size; i++) eigen_evals[i] = eigenSolver.eigenvalues()[i];
  }
  //-----------------------------------------------------------------------
  
  //Start ARPACK routines
  //---------------------------------------------------------------------------------

  do {
    
    //Interface to arpack routine
    //---------------------------
    gettimeofday(&start, NULL);    
    ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum_char, &nev_, &tol_, resid_, &nkv_, evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 2);
    
    if (info_ != 0) {
      printf("\nError in dsaupd info = %d. Exiting...\n",info_);
      arpackErrorHelpSAUPD(iparam_);
      exit(0);
    }
    
    if (ido_ == 99 || info_ == 1)
      break;
    gettimeofday(&end, NULL);  
    t_ARPACK += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

    gettimeofday(&start, NULL);
    if (ido_ == -1 || ido_ == 1) {      
      //Copy from Arpack workspace
      for(int i=0; i<n_; i++) {
	psi1_cpy[i] = *(psi1 + i);
      }
      matVec(mat, psi2_cpy, psi1_cpy);
      //Copy to Arpack workspace
      for(int i=0; i<n_; i++) {
	*(psi2 + i) = psi2_cpy[i];
      }
    }
    gettimeofday(&end, NULL);  
    t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    
    iter_cnt++;
    //if((iter_cnt)%10 == 0) printf("Arpack Iteration: %d (%e secs)\n", iter_cnt, time/(CLOCKS_PER_SEC));
  } while (99 != ido_ && iter_cnt < max_iter);
  
  //Subspace calulated sucessfully. Compute nEv eigenvectors and values
  //printf("ARPACK Finished in %e secs: iter=%04d  info=%d  ido=%d\n", time/(CLOCKS_PER_SEC), iter_cnt, info_, ido_);

  // HACK some estimates before passing to zneupd
  // Compute eigenvalues
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> _evals(nKr, 0.0);
  
  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr);
  for(int i=0; i<nKr; i++) {
    kSpace[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    zero(kSpace[i]);
    for(int j=0; j<mat_size; j++) {
      kSpace[i][j] = evecs[i*mat_size + j];
    }
  }
    
  computeEvals(mat, kSpace, residua, _evals, nEv);
  for (int i = 0; i < nEv; i++) {
    printf("EigValueEstimate[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, _evals[i].real(), _evals[i].imag(), residua[i]);
  }  
  
  printf("ARPACK Computing Eigenvlaues\n");
  ARPACK(zneupd)(&rvec_, &howmany, select_, evals_, evecs_, &n_, &sigma_, w_workev_, &bmat, &n_, spectrum_char, &nev_, &tol_, resid_,
		 &nkv_, evecs_, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_, 1, 1, 2);
  
  if (info_ == -15) {
    printf("\nError in dseupd info = %d. You likely need to\n"
	   "increase the maximum ARPACK iterations. Exiting...\n", info_);
    arpackErrorHelpSEUPD(iparam_);
    exit(0);
  } else if (info_ != 0) {
    printf("\nError in dseupd info = %d. Exiting...\n", info_);
    arpackErrorHelpSEUPD(iparam_);
  }

  // Print additional convergence information.
  if(info_ == 1){
    printf("Maximum number of iterations reached.\n");
  }
  else{
    if(info_ == 3){
      printf("Error: No shifts could be applied during implicit\n");
      printf("Error: Arnoldi update, try increasing NkV.\n");
    }
  }

  for(int i=0; i<nKr; i++) {
    for(int j=0; j<mat_size; j++) {
      kSpace[i][j] = evecs[i*mat_size + j];
    }
  }
  computeEvals(mat, kSpace, residua, _evals, nEv);

  std::vector<std::pair<Complex, int>> array(nEv);
  for(int i=0; i<nEv; i++)
    array[i] = std::make_pair(evals_[i], i);
  
  std::sort(array.begin(), array.begin() + nEv,
	    [] (const pair<Complex,int> &a,
		const pair<Complex,int> &b) {
	      return (abs(a.first) < abs(b.first)); } );
  
  
  // cleanup 
  free(ipntr_);
  free(iparam_);
  free(resid_);
  free(w_workd_);
  free(w_workl_);
  free(w_workev_);
  free(w_rwork_);
  free(select_);
  
  return 0;
}
