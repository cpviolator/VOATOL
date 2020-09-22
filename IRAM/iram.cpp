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

#define Complex complex<double>
#include "algoHelpers.h"
#include "linAlgHelpers.h"
#include "lapack.h"

int main(int argc, char **argv) {

  if (argc < 13 || argc > 13) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./irlm <nKr> <nEv> <max-restarts> <diag> <tol> <spectrum: 0=LR, 1=SR> <threads> <res_type: 0=eig, 1=schur, 2=qr> <rot_type: 0=eig, 1=schur, 2=qr> <Sym_type: 0=asym, 1=sym> <verbosity: 1=verbose, 0=quiet> <Eigen Check: 0=false, 1=true>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int max_restarts = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  bool reverse = (atoi(argv[6]) == 0 ? true : false);
  int threads = atoi(argv[7]);
  int res_type = atoi(argv[8]);
  int rot_type = atoi(argv[9]);
  bool symm = (atoi(argv[10]) == 1 ? true : false);
  bool verbose = (atoi(argv[11]) == 1 ? true : false);
  bool eigen_check = (atoi(argv[12]) == 1 ? true : false);
  omp_set_num_threads(threads);
  Eigen::setNbThreads(threads);

  
  if (!(nKr > nEv + 6)) {
    //printf("nKr=%d must be greater than nEv+6=%d\n", nKr, nEv + 6);
    //exit(0);
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
  printf("reverse = %s\n", reverse == true ? "true" :  "false");

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
  double t1 = clock();
  if(eigen_check) {
    printf("START EIGEN SOLUTION\n");
    if(symm) eigenSolver.compute(ref + ref.adjoint() + diagonal);
    else eigenSolver.compute(ref);
    double t2e = clock() - t1;
    printf("END EIGEN SOLUTION\n");
    for(int i=0; i<Nvec; i++) cout << eigenSolver.eigenvalues()[i] << " " << abs(eigenSolver.eigenvalues()[i]) << endl;
    printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  }
  //-----------------------------------------------------------------------

  // Construct objects for Arnoldi.
  //---------------------------------------------------------------------
  //Eigenvalues and their residua
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);
  std::vector<int> evals_idx(nKr, 0);

  // Krylov Space.
  std::vector<Complex*> kSpace(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  // Upper Hessenberg matrix
  std::vector<Complex*> upperHess(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    upperHess[i] = (Complex*)malloc((nKr+1)*sizeof(Complex));
    for(int j=0; j<nKr+1; j++) upperHess[i][j] = 0.0;
  }

  // Residual vector(s). Also used as temp vector(s)
  std::vector<Complex*> r(1);
  for(int i=0; i<1; i++) {
    r[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(r[i]);
  }

  // Eigen object for Arnoldi vector rotation
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH;
  Eigen::ComplexSchur<MatrixXcd> schurUH;
  MatrixXcd upperHessEigen = MatrixXcd::Zero(nKr, nKr);
  printf("START ARNOLDI SOLUTION\n");

  double epsilon = DBL_EPSILON;
  double beta = 0.0;
  //double mat_norm = 0.0;
  bool converged = false;
  int iter = 0;
  int restart_iter = 0;
  int iter_converged = 0;
  int iter_locked = 0;
  int iter_keep = 0;
  int num_converged = 0;
  int num_locked = 0;
  int num_keep = 0;
  std::vector<std::pair<Complex, int>> array(nKr);
  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int i=0; i<Nvec; i++) {
    r[0][i].real(drand48());
    r[0][i].imag(drand48());    
  }
  
  //Normalise initial source
  normalise(r[0]);
  
  //v_1
  copy(kSpace[0], r[0]);

  t1 = clock();
  
  // START IRAM
  // Implicitly restarted Arnoldi method for asymmetric eigenvalue problems
  //----------------------------------------------------------------------

  // Loop over restart iterations.
  while(restart_iter < max_restarts && !converged) {

    // p = m-k steps to get to the m-step factorisation
    for (int step = num_keep; step < nKr; step++) {
      arnoldiStep(mat, kSpace, upperHess, r, step);
      // Undo normalisation
      beta = upperHess[nKr][nKr-1].real();
      ax(1.0/beta, kSpace[nKr]);
    }
    
    iter += (nKr - num_keep);

    printf("Restart %d complete\n", restart_iter+1);
    int dim = nKr - num_locked;
    MatrixXcd Qmat = MatrixXcd::Identity(dim, dim);
    MatrixXcd Rmat = MatrixXcd::Zero(dim, dim);
    for(int k=0; k<dim; k++) {
      for(int i=0; i<dim; i++) {
	Rmat(k,i) = upperHess[k][i];
	upperHessEigen(k,i) = upperHess[k][i];
      }
    }

    if(verbose) {
      cout << upperHessEigen << endl << endl;
      cout << "symm test = " << (upperHessEigen - upperHessEigen.adjoint()).norm() << endl << endl;
      cout << upperHessEigen - upperHessEigen.adjoint() << endl << endl;
    }
    
    switch(res_type) {
    case 0:
      eigenSolverUH.compute(upperHessEigen); 
      for(int i=0; i<dim; i++)
	array[i] = std::make_pair(eigenSolverUH.eigenvalues()[i], i);
      break;
    case 1:
      schurUH.computeFromHessenberg(Rmat, Qmat);
      for(int i=0; i<dim; i++)
	array[i] = std::make_pair(schurUH.matrixT().col(i)[i], i);
      break;
    case 2:
      qrFromUpperHess(upperHessEigen, Qmat, Rmat, nKr, num_locked);
      for(int i=0; i<dim; i++)
	array[i] = std::make_pair(Rmat.col(i)[i], i);
      break;
    }

    //for(int i=0; i<dim; i++) cout << array[i].first << " " << array[i].second << endl;    
    std::sort(array.begin(), array.begin() + dim,
	      [] (const pair<Complex,int> &a,
		  const pair<Complex,int> &b) {
		return (abs(a.first) < abs(b.first)); } );   
    //for(int i=0; i<dim; i++) cout << array[i].first << " " << abs(array[i].first) << " " << array[i].second << endl;

    /*
    MatrixXcd Pmat = MatrixXcd::Zero(dim, dim);
    for (int i=0; i<dim; i++) Pmat(i,array[i].second) = 1;

    if(verbose) {
      for (int i=0; i<dim; i++) {
	for (int k=0; k<dim; k++) {
	  cout << (int)Pmat(i,k).real();
	}
	cout<<endl;
      }
    }
    */
    
    num_converged = 0;
    for(int i=num_locked; i<nKr; i++) {
      double res = 0.0;
      switch(res_type) {
      case 0:
	res = abs(beta * eigenSolverUH.eigenvectors().col(array[i].second)[dim-1]);
	break;
      case 1:
	res = abs(beta * schurUH.matrixU().col(array[i].second)[dim-1]);	
	break;
      case 2:
	res = abs(beta * Qmat.col(array[i].second)[dim-1]);
	break;
      }
      if(res < tol * abs(array[i].first)) {
	num_converged++;
      }
      if(verbose) printf("residuum[%d] = %e, cond = %e\n", i, res, tol * abs(array[i].first));
    }
    
    printf("%04d converged eigenvalues at iter %d\n", num_converged, restart_iter);

    if (num_converged >= nEv || nEv == Nvec) {
      converged = true;
    } else if (restart_iter < max_restarts) {
      Qmat.setIdentity();
      MatrixXcd sigma = MatrixXcd::Identity(nKr, nKr);
      for(int i=0; i<(dim - nEv); i++){

	if(verbose) {
	  cout << "Qmat" << endl;
	  cout << Qmat << endl;
	  cout << "UH" << endl;
	  cout << upperHessEigen << endl;
	  cout << "symm test = " << (upperHessEigen - upperHessEigen.adjoint()).norm() << endl << endl;
	}
	  
	sigma.setIdentity();
	sigma *= array[i].first;
	if(verbose) printf("Projecting out ||(%e,%e)|| = %e\n", array[i].first.real(), array[i].first.imag(), abs(array[i].first));
	upperHessEigen -= sigma;
	qriteration(upperHessEigen, Qmat, nKr);	
        upperHessEigen += sigma;

	if(verbose) {
	  cout << "Qmat" << endl;
	  cout << Qmat << endl;
	  cout << "UH" << endl;
	  cout << upperHessEigen << endl;
	  cout << "symm test = " << (upperHessEigen - upperHessEigen.adjoint()).norm() << endl << endl;
	}
	
	if(verbose) {
	  cout << Qmat << endl << endl;;
	  MatrixXcd Id = MatrixXcd::Identity(nKr, nKr);
	  cout << "Unitarity test " << (Qmat * Qmat.adjoint() - Id).norm() << endl << endl;    
	  cout << upperHessEigen << endl;
	}
      }
      
      Complex factor = Qmat(nKr-1, nEv-1);
      if(verbose) cout << upperHessEigen(nEv, nEv-1) << " " << factor << endl;
      
      rotateVecsComplex(kSpace, Qmat, num_locked, nEv, nKr);
      
      //Update residual
      caxpby(factor, kSpace[nKr], upperHessEigen(nEv, nEv-1), kSpace[nEv]);
      upperHessEigen(nEv, nEv-1).real(normalise(kSpace[nEv]));
      upperHessEigen(nEv, nEv-1).imag(0.0);
      // upperHessEigen(nEv, nEv-1).real(beta);
      // cout << upperHessEigen(nEv, nEv-1) << endl;
      // copy(kSpace[nEv], kSpace[nKr]);
      if(upperHessEigen(nEv, nEv-1) != upperHessEigen(nEv, nEv-1) || upperHessEigen(nEv, nEv-1).real() < 1e-15) {
	printf("Congratulations! You have reached an invariant subspace at iter %d, beta = %e\n", restart_iter, upperHessEigen(nEv, nEv-1).real());
	
	// Populate source with randoms.
	printf("Using random guess\n");
	for(int i=0; i<Nvec; i++) {
	  r[0][i].real(drand48());
	  r[0][i].imag(drand48());    
	}
	
	Complex alpha = 0.0;
	for(int i=0; i < nEv; i++) {
	  alpha = cDotProd(kSpace[i], r[0]);
	  upperHess[i][nEv-1] += alpha;
	  caxpy(-1.0*alpha, kSpace[i], r[0]);
	}
	
	if(verbose) {
	  // Measure orthonormality
	  for(int i=0; i < nEv; i++) {
	    alpha = cDotProd(kSpace[i], r[0]);
	    cout << "alpha = " << alpha <<endl;
	  }
	}
	
	upperHessEigen(nEv, nEv-1).real(normalise(r[0]));
	
	copy(kSpace[nEv], r[0]);
	//exit(0);
      }
      
      for(int k=0; k<dim; k++) {
	for(int i=0; i<dim; i++) {
	  upperHess[k][i] = upperHessEigen(k,i);
	}
      }
    }
    num_keep = nEv;
    restart_iter++;    
  }
  
  double t2l = clock() - t1;
  // Post computation report  
  if (!converged) {    
    printf("Arnoldi failed to compute the requested %d vectors in %d restart_steps with a %d Krylov space\n", nEv, restart_iter, nKr);
  } else {
    printf("Arnoldi computed the requested %d vectors in %d steps with a %d Krylov space in %e secs.\n", nEv, restart_iter, nKr, t2l/CLOCKS_PER_SEC);    
  }

  // Compute Eigenvalues
  eigenSolverUH.compute(upperHessEigen);
  rotateVecsComplex(kSpace, eigenSolverUH.eigenvectors(), 0, nKr, nKr);
  computeEvals(mat, kSpace, residua, evals, nKr);
  for (int i = 0; i < nEv; i++) {
    int idx = nKr - 1 - i;
    printf("EigValue[%04d]: ||(%+.8e, %+.8e)|| = %+.8e residual %.8e\n", i, evals[idx].real(), evals[idx].imag(), abs(evals[idx]), residua[idx]);
  }

  if(eigen_check) {
    for (int i = 0; i < nEv; i++) {
      int idx = nKr - 1 - i;
      int idx_e = Nvec - 1 - i;
      if(symm) {
	printf("EigenComp[%04d]: [(%+.8e, %+.8e) - (%+.8e, %+.8e)]/(%+.8e, %+.8e) = (%+.8e,%+.8e)\n", i, evals[idx].real(), evals[idx].imag(), eigenSolver.eigenvalues()[idx_e].real(), eigenSolver.eigenvalues()[idx_e].imag(), eigenSolver.eigenvalues()[idx_e].real(), eigenSolver.eigenvalues()[idx_e].imag(), (evals[idx].real() - eigenSolver.eigenvalues()[idx_e].real())/eigenSolver.eigenvalues()[idx_e].real(), (evals[idx].imag() - eigenSolver.eigenvalues()[idx_e].imag()));
      } else { 
	printf("EigenComp[%04d]: [(%+.8e, %+.8e) - (%+.8e, %+.8e)]/(%+.8e, %+.8e) = (%+.8e,%+.8e)\n", i, evals[idx].real(), evals[idx].imag(), eigenSolver.eigenvalues()[idx_e].real(), eigenSolver.eigenvalues()[idx_e].imag(), eigenSolver.eigenvalues()[idx_e].real(), eigenSolver.eigenvalues()[idx_e].imag(), (evals[idx].real() - eigenSolver.eigenvalues()[idx_e].real())/eigenSolver.eigenvalues()[idx_e].real(), (evals[idx].imag() - eigenSolver.eigenvalues()[idx_e].imag())/eigenSolver.eigenvalues()[idx_e].imag());
      }
    }
  }
}
