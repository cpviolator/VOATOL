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

#define Nvec 2048
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
#include "Eigen/Eigenvalues"
using namespace std;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

#define Complex complex<double>
#include "algoHelpers.h"
#include "linAlgHelpers.h"
#include "lapack.h"

int main(int argc, char **argv) {

  if (argc < 10 || argc > 10) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./irlm <nKr> <nEv> <max-restarts> <diag> <tol> <spectrum: 0=LM, 1=SM, 2=LR, 3=SR, 4=LI, 5=SI> <mat_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  int spectrum = atoi(argv[6]);
  bool symm = (atoi(argv[7]) == 1 ? true : false);
  bool verbose = (atoi(argv[8]) == 1 ? true : false);
  bool eigen_check = (atoi(argv[9]) == 1 ? true : false);
  int threads = atoi(std::getenv("OMP_NUM_THREADS"));
  omp_set_num_threads(threads);

  int CLOCKS_PER_CORE = (CLOCKS_PER_SEC*threads);
  
  if (!(nKr > nEv + 6)) {
    printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    exit(0);
  }

  if (nEv > Nvec || nKr > Nvec) {
    printf("nKr=%d and nEv=%d must be less than Nvec=%d\n", nKr, nEv, Nvec);
    exit(0);
  }

  printf("Mat size = %d\n", Nvec);
  printf("nKr = %d\n", nKr);
  printf("nEv = %d\n", nEv);
  printf("Restarts = %d\n", max_restarts);
  printf("diag = %e\n", diag);
  printf("tol = %e\n", tol);

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
      if(symm) mat[i][j] += conj(ref(j,i));	
      if(i == j) mat[i][j] += diag;
    }
  }
  
  // Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolver;
  std::vector<Complex> eigen_evals(Nvec, 0.0);
  double t1 = clock();
  if(eigen_check) {
    printf("START EIGEN SOLUTION\n");
    if(symm) eigenSolver.compute(ref + ref.adjoint() + diagonal);
    else eigenSolver.compute(ref + diagonal);
    double t2e = clock() - t1;
    printf("END EIGEN SOLUTION\n");
    printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_CORE);
    for(int i=0; i<Nvec; i++) eigen_evals[i] = eigenSolver.eigenvalues()[i];
  }
  //-----------------------------------------------------------------------

  // Construct objects for IRAM.
  //---------------------------------------------------------------------
  //Eigenvalues and their residua
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);

  // Krylov Space.
  std::vector<Complex*> kSpace(nKr);
  for(int i=0; i<nKr; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  // Upper Hessenberg matrix
  MatrixXcd upperHessEigen = MatrixXcd::Zero(nKr, nKr);

  // Residual vector(s). Also used as temp vector(s)
  std::vector<Complex*> r(1);
  for(int i=0; i<1; i++) {
    r[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(r[i]);
  }

  // Eigens object for Arnoldi vector rotation
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH;
  Eigen::ComplexSchur<MatrixXcd> schurUH;
  MatrixXcd Qmat = MatrixXcd::Identity(nKr, nKr);
  
  printf("START IRAM SOLUTION\n");

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
  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) {
    r[0][i].real(drand48());
    r[0][i].imag(drand48());    
  }
  
  //Place initial source in range of mat
  matVec(mat, kSpace[0], r[0]);
  copy(r[0], kSpace[0]);
  
  t1 = clock();
  
  // START IRAM
  // Implicitly restarted Arnoldi method for asymmetric eigenvalue problems
  //----------------------------------------------------------------------

  // Do the first nEv steps
  for (int step = 0; step < nEv; step++) arnoldiStepArpack(mat, kSpace, upperHessEigen, r, beta, step);
  num_keep = nEv;
  iter += nEv;
  
  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {

    for (int step = num_keep; step < nKr; step++) arnoldiStepArpack(mat, kSpace, upperHessEigen, r, beta, step);
    iter += (nKr - num_keep);

    // Construct objects for Ritz and bounds
    int dim = nKr;
    
    // Compute Ritz and bounds
    eigensolveFromUpperHess(upperHessEigen, Qmat, evals, residua, beta, nKr);    
    num_keep = nEv;    
    int nshifts = nKr - num_keep;
    
    // Emulate zngets: sort the unwanted Ritz to the start of the arrays, then
    // sort the first (nKr - nEv) bounds to be first for forward stability
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
    
    int nshifts0 = nshifts;
    for(int i=0; i<nshifts0; i++) {
      if(residua[i] <= epsilon) {
	nshifts--;
      }
    }
    
    if(nshifts == 0) {
      cout << "No shifts can be applied" << endl;
      exit(0);
    }    

    int num_keep0 = num_keep;
    iter_keep = std::min(iter_converged + (nKr - num_converged) / 2, nKr - 12);
    
    num_converged = iter_converged;
    num_keep = iter_keep;      
    nshifts = nKr - num_keep;

    printf("%04d converged eigenvalues at iter %d\n", num_converged, restart_iter);
    
    if (num_converged >= nEv || nEv == Nvec) {
      converged = true;
      // Compute Eigenvalues
      Qmat.setIdentity();      
      eigensolveFromUpperHess(upperHessEigen, Qmat, evals, residua, beta, nKr);      
      rotateVecsComplex(kSpace, Qmat, 0, nKr, nKr);
      reorder(kSpace, evals, residua, nKr, spectrum);
      computeEvals(mat, kSpace, residua, evals, nKr);      
    } else if (restart_iter < max_restarts) {
      
      //          %-------------------------------------------------%
      //          | Do not have all the requested eigenvalues yet.  |
      //          | To prevent possible stagnation, adjust the size |
      //          | of NEV.                                         |
      //          | If the size of NEV was just increased resort    |
      //          | the eigenvalues.                                |
      //          %-------------------------------------------------%

      if(num_keep0 < num_keep) {
	// Emulate zngets: sort the unwanted Ritz to the start of the arrays, then
	// sort the first (nKr - nEv) bounds to be first for forward stability
	
	// Sort to put unwanted Ritz(evals) first
	zsortc(spectrum, dim, evals, residua);
	// Sort to put smallest Ritz errors(residua) first
	zsortc(0, nshifts, residua, evals);    
      }
     
      //       %---------------------------------------------------------%
      //       | Apply the NP implicit shifts by QR bulge chasing.       |
      //       | Each shift is applied to the whole upper Hessenberg     |
      //       | matrix H.                                               |
      //       %---------------------------------------------------------%

      Qmat.setIdentity();
      MatrixXcd sigma = MatrixXcd::Identity(dim, dim);
      for(int i=0; i<nshifts; i++){
	
	if(verbose) cout << "symm test = " << (upperHessEigen - upperHessEigen.adjoint()).norm() << endl << endl;
	
	sigma.setIdentity();
	sigma *= evals[i];
	if(verbose) printf("Projecting out ||(%e,%e)|| = %e\n", evals[i].real(), evals[i].imag(), abs(evals[nshifts + i]));
	upperHessEigen -= sigma;
	qriteration(upperHessEigen, Qmat, dim);	
	upperHessEigen += sigma;
	
	if(verbose) {
	  cout << "symm test = " << (upperHessEigen - upperHessEigen.adjoint()).norm() << endl << endl;
	  MatrixXcd Id = MatrixXcd::Identity(dim, dim);
	  cout << "Unitarity test " << (Qmat * Qmat.adjoint() - Id).norm() << endl << endl;    
	}
      }
      
      rotateVecsComplex(kSpace, Qmat, 0, num_keep+1, dim);
      
      //    %-------------------------------------%
      //    | Update the residual vector:         |
      //    |    r <- sigmak*r + betak*v(:,kev+1) |
      //    | where                               |
      //    |    sigmak = (e_{kev+p}'*Q)*e_{kev}  |
      //    |    betak = e_{kev+1}'*H*e_{kev}     |
      //    %-------------------------------------%

      caxpby(upperHessEigen(num_keep, num_keep-1), kSpace[num_keep], Qmat(dim-1, num_keep-1), r[0]);
      
#if 1
      if(norm(r[0]) < epsilon) {
	printf("Congratulations! You have reached an invariant subspace at iter %d, beta = %e\n", restart_iter, norm(r[0]));
	
	printf("Using random guess\n");
	for(int i=0; i<Nvec; i++) {
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
  
  double t2l = clock() - t1;
  // Post computation report  
  if (!converged) {    
    printf("IRAM failed to compute the requested %d vectors with a %d Krylov space in %d restart_steps and %d OPs\n", nEv, nKr, restart_iter, iter);
  } else {
    printf("IRAM computed the requested %d vectors with a %d Krylov space in %d restart_steps and %d OPs in %e secs.\n", nEv, nKr, restart_iter, iter, t2l/CLOCKS_PER_CORE);    
  }

  if(eigen_check) {
    // sort the eigen eigenvalues by the requested spectrum. The wanted values
    // will appear at the end of the array. We need a dummy residua array to use
    // the sorting function.
    std::vector<double> res_dummy(Nvec, 0.0);
    zsortc(spectrum, Nvec, eigen_evals, res_dummy);
    for (int i = 0; i < nEv; i++) {
      int idx_e = Nvec - 1 - i;
      printf("EigenComp[%04d]: [(%+.8e, %+.8e) - (%+.8e, %+.8e)]/(%+.8e, %+.8e) = "
	     "(%+.8e,%+.8e)\n", i,
	     evals[i].real(), evals[i].imag(),
	     eigen_evals[idx_e].real(), eigen_evals[idx_e].imag(),
	     eigen_evals[idx_e].real(), eigen_evals[idx_e].imag(),
	     (evals[i].real() - eigen_evals[idx_e].real())/eigen_evals[idx_e].real(),
	     (evals[i].imag() - eigen_evals[idx_e].imag())/eigen_evals[idx_e].imag());
    }
  } else {
    // "reorder" places the wanted eigenvalues and eigenvectors at the start of the
    // array
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: ||(%+.8e, %+.8e)|| = %+.8e residual %.8e\n", i, evals[i].real(), evals[i].imag(), abs(evals[i]), residua[i]);
    }
  }  
}
