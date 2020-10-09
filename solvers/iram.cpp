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
#include "algoHelpers.h"
#include "linAlgHelpers.h"
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
  double t_QR = 0.0;
  double t_EV = 0.0;  

  // START init
  //---------------------------------------------------------  
  gettimeofday(&start, NULL);  
  if (argc < 12 || argc > 12) {
    cout << "./irlm <mat_size> <nKr> <nEv> <nConv> <max-restarts> <diag> <tol> <spectrum: 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI> <mat_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>" << endl;
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

  // Construct objects for IRAM.
  //---------------------------------------------------------------------
  //Eigenvalues and their residua
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);

  // Krylov Space.
  std::vector<Complex*> kSpace(nKr);
  for(int i=0; i<nKr; i++) {
    kSpace[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    zero(kSpace[i]);
  }

  // Upper Hessenberg matrix
  MatrixXcd upperHessEigen = MatrixXcd::Zero(nKr, nKr);

  // Residual vector(s). Also used as temp vector(s)
  std::vector<Complex*> r(1);
  for(int i=0; i<1; i++) {
    r[i] = (Complex*)malloc(mat_size*sizeof(Complex));
    zero(r[i]);
  }

  // Eigens object for Arnoldi vector rotation and QR shifts
  MatrixXcd Qmat = MatrixXcd::Identity(nKr, nKr);
  MatrixXcd sigma = MatrixXcd::Identity(nKr, nKr);
 
  double epsilon = DBL_EPSILON;
  double epsilon23 = pow(epsilon, 2.0/3.0);
  double beta = 0.0;
  bool converged = false;
  int iter = 0;
  int restart_iter = 0;
  int iter_converged = 0;
  int iter_keep = 0;
  int num_converged = 0;
  int num_keep = 0;  
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


  // START compute 
  //---------------------------------------------------------
  gettimeofday(&start, NULL);  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<mat_size; i++) {
    r[0][i].real(drand48());
    r[0][i].imag(drand48());    
  }
  
  //Place initial source in range of mat
  matVec(mat, kSpace[0], r[0]);
  copy(r[0], kSpace[0]);
  
  // START IRAM
  // Implicitly restarted Arnoldi method for asymmetric eigenvalue problems
  printf("START IRAM SOLUTION\n");
  //----------------------------------------------------------------------

  // Do the first nEv steps
  for (int step = 0; step < nEv; step++) arnoldiStep(mat, kSpace, upperHessEigen, r, beta, step);
  num_keep = nEv;
  iter += nEv;
  gettimeofday(&end, NULL);  
  t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
  
  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {

    gettimeofday(&start, NULL); 
    for (int step = num_keep; step < nKr; step++) arnoldiStep(mat, kSpace, upperHessEigen, r, beta, step);
    iter += (nKr - num_keep);
    gettimeofday(&end, NULL);  
    t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    
    // Construct objects for Ritz and bounds
    int dim = nKr;
    
    // Compute Ritz and bounds
    gettimeofday(&start, NULL);  
    //eigensolveFromUpperHess(upperHessEigen, Qmat, evals, residua, beta, nKr);
    qrFromUpperHess(upperHessEigen, Qmat, evals, residua, beta, nKr, tol/10);
    gettimeofday(&end, NULL);  
    t_EV += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    
    num_keep = nEv;    
    int nshifts = nKr - num_keep;
    
    // Emulate zngets: sort the unwanted Ritz to the start of the arrays, then
    // sort the first (nKr - nEv) bounds to be first for forward stability
    gettimeofday(&start, NULL); 
    // Sort to put unwanted Ritz(evals) first
    zsortc(spectrum, dim, evals, residua);    
    // Sort to put smallest Ritz errors(residua) first
    zsortc(0, nshifts, residua, evals);
    
    // Convergence test
    iter_converged = 0;
    for(int i=0; i<nEv; i++) {
      int idx = dim - 1 - i;
      double rtemp = std::max(epsilon23, abs(evals[idx]));
      if(residua[idx] < tol * rtemp) {
	iter_converged++;
	if(verbose) printf("residuum[%d] = %e, cond = %e\n", i, residua[idx], tol * abs(evals[idx]));
      } else {
	break;
      }
    }    

    //       %---------------------------------------------------------%
    //       | Count the number of unwanted Ritz values that have zero |
    //       | Ritz estimates. If any Ritz estimates are equal to zero |
    //       | then a leading block of H of order equal to at least    |
    //       | the number of Ritz values with zero Ritz estimates has  |
    //       | split off. None of these Ritz values may be removed by  |
    //       | shifting. Decrease NP the number of shifts to apply. If |
    //       | no shifts may be applied, then prepare to exit          |
    //       %---------------------------------------------------------%
    
    int num_keep0 = num_keep;
    iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - 12);
    
    num_converged = iter_converged;
    num_keep = iter_keep;      
    nshifts = nKr - num_keep;

    printf("%04d converged eigenvalues at iter %d\n", num_converged, restart_iter);

    int nshifts0 = nshifts;
    for(int i=0; i<nshifts0; i++) {
      if(residua[i] <= epsilon) {
	nshifts--;
      }
    }
    
    if(nshifts == 0 && num_converged < nEv) {
      cout << "No shifts can be applied" << endl;
      exit(0);
    }    
    
    gettimeofday(&end, NULL);
    t_sort += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
    
    if (num_converged >= nConv || nEv == mat_size) {
      converged = true;
      // Compute Eigenvalues
      gettimeofday(&start, NULL);
      Qmat.setIdentity();
      qrFromUpperHess(upperHessEigen, Qmat, evals, residua, beta, nKr, 1e-15);
      gettimeofday(&end, NULL);  
      t_EV += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
      
      gettimeofday(&start, NULL);
      rotateVecsComplex(kSpace, Qmat, 0, nKr, nKr);
      reorder(kSpace, evals, residua, nKr, spectrum);
      computeEvals(mat, kSpace, residua, evals, nKr);
      gettimeofday(&end, NULL);  
      t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
      
    } else if (restart_iter < max_restarts) {
      
      //          %-------------------------------------------------%
      //          | Do not have all the requested eigenvalues yet.  |
      //          | To prevent possible stagnation, adjust the size |
      //          | of NEV.                                         |
      //          | If the size of NEV was just increased resort    |
      //          | the eigenvalues.                                |
      //          %-------------------------------------------------%

      if(num_keep0 < num_keep) {
	gettimeofday(&start, NULL); 
	// Emulate zngets: sort the unwanted Ritz to the start of the arrays, then
	// sort the first (nKr - nEv) bounds to be first for forward stability	
	// Sort to put unwanted Ritz(evals) first
	zsortc(spectrum, dim, evals, residua);
	// Sort to put smallest Ritz errors(residua) first
	zsortc(0, nshifts, residua, evals);
	gettimeofday(&end, NULL);  
	t_sort += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
      }
     
      //       %---------------------------------------------------------%
      //       | Apply the NP implicit shifts by QR bulge chasing.       |
      //       | Each shift is applied to the whole upper Hessenberg     |
      //       | matrix H.                                               |
      //       %---------------------------------------------------------%

      gettimeofday(&start, NULL); 
      Qmat.setIdentity();
      sigma.setIdentity();      
      for(int i=0; i<nshifts; i++){	
	sigma.setIdentity();
	sigma *= evals[i];
	upperHessEigen -= sigma;
	qriteration(upperHessEigen, Qmat, dim, tol/10);	
	upperHessEigen += sigma;	
      }
      gettimeofday(&end, NULL);  
      t_QR += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

      gettimeofday(&start, NULL); 
      rotateVecsComplex(kSpace, Qmat, 0, num_keep+1, dim);
      
      //    %-------------------------------------%
      //    | Update the residual vector:         |
      //    |    r <- sigmak*r + betak*v(:,kev+1) |
      //    | where                               |
      //    |    sigmak = (e_{kev+p}'*Q)*e_{kev}  |
      //    |    betak = e_{kev+1}'*H*e_{kev}     |
      //    %-------------------------------------%

      caxpby(upperHessEigen(num_keep, num_keep-1), kSpace[num_keep], Qmat(dim-1, num_keep-1), r[0]);
      gettimeofday(&end, NULL);  
      t_compute += ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
 #if 1
      if(norm(r[0]) < epsilon) {
	printf("Congratulations! You have reached an invariant subspace at iter %d, beta = %e\n", restart_iter, norm(r[0]));
	
	printf("Using random guess\n");
	for(int i=0; i<mat_size; i++) {
	  kSpace[num_keep][i].real(drand48());
	  kSpace[num_keep][i].imag(drand48());    
	}
	
	Complex alpha = 0.0;
	for(int i=0; i < num_keep; i++) {
	  alpha = cDotProd(kSpace[i], kSpace[num_keep]);
	  upperHessEigen(i,num_keep-1) += alpha;
	  caxpy(-1.0*alpha, kSpace[i], kSpace[num_keep]);
	}
	
	if(verbose) {
	  // Measure orthonormality
	  for(int i=0; i < num_keep; i++) {
	    alpha = cDotProd(kSpace[i], kSpace[num_keep]);
	    cout << "alpha = " << alpha <<endl;
	  }
	}
	upperHessEigen(num_keep, num_keep-1).real(normalise(kSpace[num_keep]));
	caxpby(upperHessEigen(num_keep, num_keep-1), kSpace[num_keep], Qmat(dim-1, num_keep-1), r[0]);
	nEv -= 1;
      }
#endif
      
    }
    restart_iter++;    
  }

  gettimeofday(&total_end, NULL);  
  t_total = ((total_end.tv_sec  - total_start.tv_sec) * 1000000u + total_end.tv_usec - total_start.tv_usec) / 1.e6;
  
  // Post computation report  
  if (!converged) {    
    printf("IRAM failed to compute the requested %d vectors with with a %d search space and a %d Krylov space in %d restart_steps and %d OPs.\n", nConv, nEv, nKr, restart_iter, iter);
  } else {
    printf("IRAM computed the requested %d vectors with a %d search space and a %d Krylov space in %d restart_steps and %d OPs in %e secs.\n", nConv, nEv, nKr, restart_iter, iter, (t_compute + t_sort + t_EV + t_QR));

    for (int i = 0; i < nConv; i++) {
      printf("EigValue[%04d]: ||(%+.8e, %+.8e)|| = %+.8e residual %.8e\n", i, evals[i].real(), evals[i].imag(), abs(evals[i]), residua[i]);
    }
    
    if(eigen_check) {
      // sort the eigen eigenvalues by the requested spectrum. The wanted values
      // will appear at the end of the array. We need a dummy residua array to use
      // the sorting function.
      std::vector<double> res_dummy(mat_size, 0.0);
      zsortc(spectrum, mat_size, eigen_evals, res_dummy);
      for (int i = 0; i < nConv; i++) {
	int idx_e = mat_size - 1 - i;
	printf("EigenComp[%04d]: [(%+.8e, %+.8e) - (%+.8e, %+.8e)]/(%+.8e, %+.8e) = "
	       "(%+.8e,%+.8e)\n", i,
	       evals[i].real(), evals[i].imag(),
	       eigen_evals[idx_e].real(), eigen_evals[idx_e].imag(),
	       eigen_evals[idx_e].real(), eigen_evals[idx_e].imag(),
	       (evals[i].real() - eigen_evals[idx_e].real())/eigen_evals[idx_e].real(),
	       (evals[i].imag() - eigen_evals[idx_e].imag())/eigen_evals[idx_e].imag());
      }
    }
  }
  
  cout << "Timings:" << endl;
  if(eigen_check) cout << "Eigen = " << t_eigen << endl;
  cout << "init = " << t_init << endl;
  cout << "compute = " << t_compute << endl;
  cout << "sort = " << t_sort << endl;
  cout << "EV = " << t_EV << endl;
  cout << "QR = " << t_QR << endl;
  cout << "missing = " << (t_total) << " - " << (t_compute + t_init + t_sort + t_EV + t_QR + t_eigen) << " = " << (t_total - (t_compute + t_init + t_sort + t_EV + t_QR + t_eigen)) << " ("<<(100*((t_total - (t_compute + t_init + t_sort + t_EV + t_QR + t_eigen))))/t_total<<"%)" << endl;
  
}
