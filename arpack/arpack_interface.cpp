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

#include <arpack_interface.h>

#define Nvec 128
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
Eigen::IOFormat CleanFmt(16, 0, ", ", "\n", "[", "]");
Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver;

bool verbose = false;

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "lapack.h"
#include "algoHelpers.h"

//====================================================================================
//
// Thu Aug 23 Dean Howarth
//  
// ARPACK interafce for AdS2-Lattice.
// 
//====================================================================================

int main(int argc, char **argv) {

  cout << std::setprecision(16);
  cout << scientific;  
  //Define the problem
  if (argc < 7 || argc > 7) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./arpack <nKr> <nEv> <max-restarts> <diag> <tol> <hermitian>" << endl;
    cout << "./arpack 60 20 200 100 1e-12 1 " << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  bool hermitian = atoi(argv[6]) == 1 ? true : false;

  // Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  MatrixXcd diagonal = MatrixXcd::Identity(Nvec, Nvec);
  diagonal *= diag;
  
  //Copy Eigen ref matrix.
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    for(int j=0; j<Nvec; j++) {
      mat[i][j] = ref(i,j);
      if(hermitian) mat[i][j] += conj(ref(j,i));	
      if(i == j) mat[i][j] += diag;
    }
  }
  
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();
  Eigen::ComplexEigenSolver<MatrixXcd> eigensolverRef;
  if(hermitian) eigensolverRef.compute(ref + ref.adjoint() + diagonal);
  else eigensolverRef.compute(ref); 
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  cout << eigensolverRef.eigenvalues() << endl;
  //-----------------------------------------------------------------------

  
  //Construct parameters and memory allocation
  //------------------------------------------  
  // all FORTRAN communication uses underscored 
  int ido_ = 0;
  int info_ = 1;
  int *ipntr_ = (int*)malloc(14*sizeof(int));
  int *iparam_ = (int*)malloc(11*sizeof(int));
  int n_    = Nvec,
    nev_    = nEv,
    nkv_    = nKr,
    ldv_    = Nvec,
    lworkl_ = (3 * nKr * nKr + 5 * nKr) * 2,
    rvec_   = 1;
  int max_iter = max_restarts * (nKr - nEv) + nEv;
  cout << max_iter << endl;
  
  double tol_ = tol;

  //ARPACK workspace
  Complex sigma_ = 0.0;
  Complex *resid_ = (Complex *) malloc(ldv_*sizeof(Complex));
  Complex *w_workd_ = (Complex *) malloc(3*ldv_*sizeof(Complex));
  Complex *w_workl_ = (Complex *) malloc(lworkl_*sizeof(Complex)); 
  Complex *w_workev_= (Complex *) malloc(2*nkv_*sizeof(Complex));
  double *w_rwork_ = (double *) malloc(nkv_*sizeof(double));    
  int *select_ = (int*)malloc(nkv_*sizeof(int));
  
  Complex *evecs = (Complex *) malloc(nkv_*ldv_*sizeof(Complex));
  Complex *evals = (Complex *) malloc(nev_   *sizeof(Complex));

  Complex one(1.0,0.0);
  
  for(int n=0; n<nkv_; n++) {
    evals[n] = 0;
    for(int i=0; i<n_; i++) {
      evecs[n*n_ + i] = 0;
      if(n==0) resid_[i] = drand48();
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
  info_       = 0;
  iparam_[0]  = 1;
  iparam_[2]  = max_iter;
  iparam_[3]  = 1;
  iparam_[6]  = 1;
  iparam_[1]  = 0;
  iparam_[4]  = 0;
  iparam_[5]  = 0;
  
  //ARPACK problem type to be solved
  char howmany = 'A';
  char bmat = 'I';
  char spectrum[3] = {'S','M'};
  int iter_cnt= 0;

  //Start ARPACK routines
  //---------------------------------------------------------------------------------
 
  Complex *psi1;
  Complex *psi2;

  Complex *psi1_cpy = (Complex*)malloc(n_*sizeof(Complex));
  Complex *psi2_cpy = (Complex*)malloc(n_*sizeof(Complex));
  
  psi1 = w_workd_;
  psi2 = w_workd_ + n_;
  
  double time = 0.0;;
  do {
    
    t1 = -((double)clock());
    
    //Interface to arpack routines
    //----------------------------
    ARPACK(znaupd)(&ido_, &bmat, &n_, spectrum, &nev_, &tol_, resid_, &nkv_, evecs, &n_, iparam_, ipntr_, w_workd_, w_workl_, &lworkl_, w_rwork_, &info_);
    
    if (info_ != 0) {
      printf("\nError in dsaupd info = %d. Exiting...\n",info_);
      arpackErrorHelpSAUPD(iparam_);
      exit(0);
    }
    
    if (ido_ == 99 || info_ == 1)
      break;
    
    if (ido_ == -1 || ido_ == 1) {

      //Copy from Arpack workspace
      for(int i=0; i<n_; i++) {
	psi1_cpy[i] = *(psi1 + i);
      }

      //cout << "Apply Mat Vec input" << endl;
      //for(int xx=0; xx<10; xx++) cout << psi1_cpy[xx] << endl;
      matVec(mat, psi2_cpy, psi1_cpy);
      //cout << "Apply Mat Vec output" << endl;
      //for(int xx=0; xx<10; xx++) cout << psi2_cpy[xx] << endl;
      //cout << endl;
      //Copy to Arpack workspace
      for(int i=0; i<n_; i++) {
	*(psi2 + i) = psi2_cpy[i];
      }
    }

    for(int q = 0; q<nKr; q++) {
      //cout << iter_cnt << " " << q << " " << w_workl_[nKr*nKr + q] << " " << w_workl_[nKr*nKr + nKr + q] << endl;
    }

    //  for(int q = 0; q<nKr*nKr; q++) {
    //   printf("(%+.3e,%+.3e) ", w_workl_[nKr*nKr+2*nKr + q].real(), w_workl_[nKr*nKr+2*nKr+1 + q].imag());
    //   if((q+1)%nKr == 0) cout << endl;
    // }
    // cout << endl;
    // cout << endl;
    
    t1 += clock();
    time += t1;
    printf("Arpack Iteration: %d (%e secs)\n", iter_cnt, time/(CLOCKS_PER_SEC));
    iter_cnt++;
    
  } while (99 != ido_ && iter_cnt < max_iter);
  
  //Subspace calulated sucessfully. Compute nEv eigenvectors and values
  printf("ARPACK Finished in %e secs: iter=%04d  info=%d  ido=%d\n", time/(CLOCKS_PER_SEC), iter_cnt, info_, ido_);

  // HACK some estimates before passing to zneupd
  // Compute eigenvalues

  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals__(nKr, 0.0);

  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
    for(int j=0; j<Nvec; j++) {
      kSpace[i][j] = evecs[i*Nvec + j];
    }
  }
  
  computeEvals(mat, kSpace, residua, evals__, nEv);
  
  for (int i = 0; i < nEv; i++) {
    printf("EigValueEstimate[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals__[i].real(), evals__[i].imag(), residua[i]);
  }  

  /*
  for (int i = 0; i < nEv; i++) {
    int idx = i;
    printf("EigenCompEstimates[%04d]: (%+.16e,%+.16e) diff = (%+.16e,%+.16e)\n", i,
	   eigensolverRef.eigenvalues()[idx].real(), eigensolverRef.eigenvalues()[idx].imag(),
	   ((evals[i] - eigensolverRef.eigenvalues()[idx]).real()/eigensolverRef.eigenvalues()[idx]).real(),
	   ((evals[i] - eigensolverRef.eigenvalues()[idx]).imag()/eigensolverRef.eigenvalues()[idx]).imag()
	   );
  }
  */
  
  printf("ARPACK Computing Eigenvlaues\n");
  ARPACK(zneupd)(&rvec_, &howmany, select_, evals_, evecs_, &n_, &sigma_, w_workev_, &bmat, &n_, spectrum, &nev_, &tol_, resid_,
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

  //Print Evalues  
  for(int i=0; i<nev_; i++){    
    printf("EigVal[%04d]: (%+.16e, %+.16e)\n", i,
	   evals_[i].real(), evals[i].imag());
    
  }
  /*
  //Compare Evalues  
  for(int i=0; i<nev_; i++){
    printf("EigenComp[%04d]: (%+.16e,%+.16e) diff = (%+.16e,%+.16e)\n", i,
	   evals_[i].real(), evals[i].imag(),
	   ((evals_[i] - eigensolverRef.eigenvalues()[i])/eigensolverRef.eigenvalues()[i]).real(),
	   ((evals_[i] - eigensolverRef.eigenvalues()[i])/eigensolverRef.eigenvalues()[i]).imag()
	   );
  }
  */
  
  t1 += clock();
  //printf("\n*************************************************\n");
  //printf("%d Eigenvalues of hamiltonian computed in: %f sec\n", nev_, t1/(CLOCKS_PER_SEC));
  //printf("Total time spent in ARPACK: %f sec\n", (time+t1)/(CLOCKS_PER_SEC));
  //printf("*************************************************\n");
  
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

