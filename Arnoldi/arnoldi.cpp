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
#define Complex complex<double>

bool verbose = false;

#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 7 || argc > 7) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./arnoldi <nKr> <nEv> <check-interval> <tol> <threads> <QR_type: 0=eig, 1=schur, 2=qr>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int check_interval = atoi(argv[3]);
  double tol = atof(argv[4]);
  int threads = atoi(argv[5]);
  int QR_type = atoi(argv[6]);

  omp_set_num_threads(threads);
  Eigen::setNbThreads(threads);

  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  
  //Copy Eigen ref matrix.
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    for(int j=0; j<Nvec; j++) {
      mat[i][j] = ref(i,j);
    }
  }
    
  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  //Eigen::ComplexEigenSolver<MatrixXcd> eigenSolver(ref);
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //-----------------------------------------------------------------------

  //Construct objects for Arnoldi.
  //---------------------------------------------------------------------
  //Eigenvalues and their residua
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);
  std::vector<std::pair<Complex, int>> array(nKr);
  
  //Krylov Space.
  std::vector<Complex*> kSpace(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  //Upper Hessenberg matrix
  std::vector<Complex*> upperHess(nKr+1);
  for(int i=0; i<nKr+1; i++) {
    upperHess[i] = (Complex*)malloc((nKr+1)*sizeof(Complex));
    for(int j=0; j<nKr+1; j++) upperHess[i][j] = 0.0;
  }

  //Residual vector(s). Also used as temp vector(s)
  std::vector<Complex*> r(1);
  for(int i=0; i<1; i++) {
    r[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(r[i]);
  }

  printf("START ARNOLDI SOLUTION\n");

  bool convergence = false;
  int num_converged = 0;
  double mat_norm = 0;  
  
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
  
  // START ARNOLDI
  // ARNOLDI Method for Asymmetric Eigenvalue Problems
  //-------------------------------------------------
    // Eigen objects for Arnoldi vector rotation
  Eigen::ComplexEigenSolver<MatrixXcd> eigenSolverUH;  
  Eigen::ComplexSchur<MatrixXcd> schurUH;
  
  int j=0;
  while(!convergence && j < nKr) {
    
    arnoldiStep(mat, kSpace, upperHess, r, j);
    
    if((j+1)%check_interval == 0) {
      int dim = j+1;
      MatrixXcd Qmat = MatrixXcd::Identity(dim, dim);
      MatrixXcd Rmat = MatrixXcd::Zero(dim, dim);
      MatrixXcd upperHessEigen = MatrixXcd::Zero(dim, dim);      
      for(int k=0; k<dim; k++) {
	for(int i=0; i<dim; i++) {
	  Rmat(k,i) = upperHess[k][i];
	  upperHessEigen(k,i) = upperHess[k][i];
	}
      }      
      double mat_norm = 0.0;
      switch(QR_type) {
      case 0:
	mat_norm = upperHessEigen.norm();
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
	qrFromUpperHess(upperHessEigen, Qmat, Rmat, dim, 0);
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
      
      double beta = upperHess[dim][dim-1].real();
      num_converged = 0;
      for(int i=0; i<dim; i++) {
	double res = 0.0;
	switch(QR_type) {
	case 0:
	  res = abs(beta * eigenSolverUH.eigenvectors().col(i)[dim-1]);
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
	//printf("residuum[%d] = %e, cond = %e\n", i, res, tol * abs(array[i].first));
      }
      
      printf("%04d converged eigenvalues at iter %d\n", num_converged, j);
      if (num_converged >= nEv) convergence = true;      
    }
    j++;
  }
    
  double t2l = clock() - t1;
  
  // Post computation report  
  if (!convergence) {    
    printf("Arnoldi failed to compute the requested %d vectors with a %d Krylov space\n", nEv, nKr);
  } else {
    printf("Arnoldi computed the requested %d vectors in %d steps in %e secs.\n", nEv, j, t2l/CLOCKS_PER_SEC);
    
    // Compute Eigenvalues
    int dim = j;
    MatrixXcd Qmat = MatrixXcd::Identity(dim, dim);
    MatrixXcd Rmat = MatrixXcd::Zero(dim, dim);
    MatrixXcd upperHessEigen = MatrixXcd::Zero(dim, dim);      
    for(int k=0; k<dim; k++) {
      for(int i=0; i<dim; i++) {
	Rmat(k,i) = upperHess[k][i];
	upperHessEigen(k,i) = upperHess[k][i];
      }
    }

    switch(QR_type) {
    case 0:
      eigenSolverUH.compute(upperHessEigen);
      cout << eigenSolverUH.eigenvectors() << endl << endl;
      for(int i=0; i<dim; i++)
	array[i] = std::make_pair(eigenSolverUH.eigenvalues()[i], i);
      break;
    case 1:
      schurUH.computeFromHessenberg(Rmat, Qmat);
      cout << schurUH.matrixU() << endl << endl;
      for(int i=0; i<dim; i++)
	array[i] = std::make_pair(schurUH.matrixT().col(i)[i], i);
      break;
    case 2:
      qrFromUpperHess(upperHessEigen, Qmat, Rmat, dim, 0);
      cout << Qmat << endl << endl;
      for(int i=0; i<dim; i++)
	array[i] = std::make_pair(Rmat.col(i)[i], i);
      break;
    }

    for(int i=0; i<dim; i++) cout << array[i].first << " " << array[i].second << endl; 
    std::sort(array.begin(), array.begin() + dim,
	      [] (const pair<Complex,int> &a,
		  const pair<Complex,int> &b) {
		return (abs(a.first) > abs(b.first)); } );   
    for(int i=0; i<dim; i++) cout << setprecision(16) << array[i].first << " " << abs(array[i].first) << " " << array[i].second << setprecision(6) << endl;

    MatrixXcd Pmat = MatrixXcd::Zero(j, j);
    for (int i=0; i<j; i++) Pmat(i,array[i].second) = 1;

    switch(QR_type) {
    case 0:
      cout << eigenSolverUH.eigenvectors()*Pmat << endl << endl;
      break;
    case 1:
      cout << Pmat.adjoint()*schurUH.matrixU() << endl << endl;
      break;
    case 2:
      cout << Pmat.adjoint()*Qmat << endl << endl;
      break;
    }

    
    switch(QR_type) {
    case 0:
      rotateVecsComplex(kSpace, eigenSolverUH.eigenvectors()*Pmat, 0, j, j);
      break;
    case 1:
      // Compute permutation
      rotateVecsComplex(kSpace, schurUH.matrixU(), 0, j, j);
      break;
    case 2:
      // Compute permutation
      rotateVecsComplex(kSpace, Qmat, 0, j, j);
      //rotateVecsComplex(kSpace, Qmat*Pmat, 0, j, j);
      //rotateVecsComplex(kSpace, Pmat*Qmat, 0, j, j);
      //rotateVecsComplex(kSpace, Pmat*Qmat*Pmat, 0, j, j);
      break;
    }

    for (int i=0; i<j; i++) {
      for (int k=0; k<j; k++) {
	cout << Pmat(i,k);
      }
      cout<<endl;
    }
    cout<<endl;
    for (int i=0; i<j; i++) {
      for (int k=0; k<j; k++) {
	cout << Pmat(k,i);
      }
      cout<<endl;
    }

    //for (int i=0; i<j; i++) normalise(kSpace[i]);
    computeEvals(mat, kSpace, residua, evals, j);
    for (int i = 0; i < nEv; i++) {
      int idx = j - 1 - i;
      printf("EigValue[%04d]: ||(%+.16e, %+.16e)|| = %+.16e residual %.16e\n", i, evals[idx].real(), evals[idx].imag(), abs(evals[idx]), residua[idx]);
    }
    
    for (int i = 0; i < nEv; i++) {
      int idx = j - 1 - i;
      int idx_e = Nvec - 1 - i;
      //printf("EigenComp[%04d]: (%+.8e,%+.8e)\n", i, (evals[idx].real() - eigenSolver.eigenvalues()[idx_e].real())/eigenSolver.eigenvalues()[idx_e].real(), (evals[idx].imag() - eigenSolver.eigenvalues()[idx_e].imag())/eigenSolver.eigenvalues()[idx_e].imag());
    }
  }  
}
