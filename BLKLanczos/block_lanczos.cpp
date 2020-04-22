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

#define Complex complex<double>
#include "linAlgHelpers.h"
#include "algoHelpers.h"

int main(int argc, char **argv) {

  //Define the problem
  if (argc < 12 || argc > 12) {
    cout << "Built for matrix size " << Nvec << endl;
    cout << "./lanczos <nKr> <nEv> <check-interval> <diag> <tol> <amin> <amax> <polydeg> <block_size> <threads> <Real=0, Complex=1>" << endl;
    exit(0);
  }
  
  int nKr = atoi(argv[1]);
  int nEv = atoi(argv[2]);
  int check_interval = atoi(argv[3]);
  double diag = atof(argv[4]);
  double tol = atof(argv[5]);
  double a_min = atof(argv[6]);
  double a_max = atof(argv[7]);
  double poly_deg = atof(argv[8]);
  int block_size = atoi(argv[9]);
  int threads = atoi(argv[10]);
  int reCom = atoi(argv[11]);
  
  omp_set_num_threads(threads);
  Eigen::setNbThreads(threads);

  //Construct a matrix using Eigen.
  //---------------------------------------------------------------------  
  MatrixXcd ref = MatrixXcd::Random(Nvec, Nvec);
  if(reCom == 0) {
    for(int i=0; i<Nvec; i++) 
      for(int j=0; j<Nvec; j++) ref(i,j).imag(0.0);
  }
  
  //Copy Eigen ref matrix.
  Complex **mat = (Complex**)malloc(Nvec*sizeof(Complex*));
  for(int i=0; i<Nvec; i++) {
    mat[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    ref(i,i) = Complex(diag,0.0);
    mat[i][i] = ref(i,i);
    
    for(int j=0; j<i; j++) {
      mat[j][i] = ref(i,j);
      mat[i][j] = conj(ref(i,j));
      ref(j,i)  = conj(ref(i,j));
    }
  }

  //Eigensolve the matrix using Eigen, use as a reference.
  //---------------------------------------------------------------------  
  printf("START EIGEN SOLUTION\n");
  double t1 = clock();  
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigenSolverRef(ref);
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigenSolverTD;
  double t2e = clock() - t1;
  printf("END EIGEN SOLUTION\n");
  printf("Time to solve problem using Eigen = %e\n", t2e/CLOCKS_PER_SEC);
  //-----------------------------------------------------------------------

  //Construct objects for Lanczos.
  //---------------------------------------------------------------------
  //Eigenvalues and their residuals
  std::vector<double> residua(nKr, 0.0);
  std::vector<Complex> evals(nKr, 0.0);
  
  //Ritz vectors and Krylov Space. The eigenvectors will be stored here.
  std::vector<Complex*> kSpace(nKr+block_size);
  for(int i=0; i<nKr+block_size; i++) {
    kSpace[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(kSpace[i]);
  }

  //Symmetric block tridiagonal matrix
  std::vector<Complex> alpha(nKr * block_size, 0.0);
  std::vector<Complex>  beta(nKr * block_size, 0.0);
  
  //Residual vectors. Also used as a temp vector
  std::vector<Complex*> r(block_size);
  for(int i=0; i<block_size; i++) {
    r[i] = (Complex*)malloc(Nvec*sizeof(Complex));
    zero(r[i]);
  }
  
  printf("START LANCZOS SOLUTION\n");

  bool convergence = false;
  int num_converged = 0;
  double mat_norm = 0;  
  
  // Populate source with randoms.
  printf("Using random guess\n");
  for(int b=0; b<block_size; b++) {
    for(int i=0; i<Nvec; i++) {
      r[b][i].real(drand48());
      //r[b][i].imag(drand48());
      //printf("elem [%d][%d] = (%e,%e)\n", b, i, r[b][i].real(), r[b][i].imag());
    }
  }
  
  //Gram-Schmidt initial sources
  gramSchmidt(r);
  orthoCheck(r, block_size, true);
  
  for(int b=0; b<block_size; b++) copy(kSpace[b], r[b]);

  t1 = clock();
  
  // START BLOCK LANCZOS
  // Block Lanczos Method for Symmetric Eigenvalue Problems
  //-------------------------------------------------
  
  int j=0;
  while(!convergence && j < nKr) {

    printf("iter %d start\n", j);
    blockLanczosStep(mat, kSpace, beta, alpha, r, -1, j, block_size, a_min, a_max, poly_deg);  
    printf("iter %d step done\n", j);
    
    if(j%check_interval == 0) {
      
      //Compute the Tridiagonal matrix T_k
      MatrixXcd triDiag = MatrixXcd::Zero(j+block_size, j+block_size);
      
      // Add the alpha/beta blocks
      int blocks = (j + block_size) / block_size;
      int block_data_length = block_size * block_size;
      int idx = 0;
      for(int i=0; i<blocks; i++) {
	for(int b=0; b<block_size; b++) {
	  for(int c=0; c<block_size; c++) {
	    idx = b*block_size + c;
	    triDiag(i*block_size + b, i*block_size + c) = alpha[i*block_data_length + idx];
	  }
	}
	if(i < blocks-1) {
	  for(int b=0; b<block_size; b++) {
	    for(int c=0; c<b+1; c++) {
	      idx = b*block_size + c;
	      // Sub diag
	      triDiag((i+1)*block_size + c, i*block_size + b) = beta[i*block_data_length + idx];
	      // Super diag
	      triDiag(i*block_size + b, (i+1)*block_size + c) = conj(beta[i*block_data_length + idx]);	      
	    }
	  }
	}
      }

      if(j<16) std::cout << triDiag << std::endl << std::endl;
      
      //Eigensolve the T_k matrix
      eigenSolverTD.compute(triDiag);

      if(j>=nEv) {
	// Compute eigenvalues
	//Ritz vectors and Krylov Space. The eigenvectors will be stored here.
	std::vector<Complex*> kSpaceCopy(nKr+block_size);
	for(int i=0; i<nKr+block_size; i++) {
	  kSpaceCopy[i] = (Complex*)malloc(Nvec*sizeof(Complex));
	  copy(kSpaceCopy[i], kSpace[i]);
	} 
	rotateVecsComplex(kSpaceCopy, eigenSolverTD.eigenvectors(), 0, j+block_size, j+block_size);
	computeEvals(mat, kSpaceCopy, residua, evals, nEv);
	for (int i = 0; i < nEv; i++) {
	  printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(), residua[i]);
	}
	
	for (int i = 0; i < nEv; i++) {
	  printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigenSolverRef.eigenvalues()[i])/eigenSolverRef.eigenvalues()[i]); 
	}      
      }
      //std::cout << eigenSolverTD.eigenvalues() << std::endl << std::endl;
      
      // mat_norm and rediua are updated.
      for (int i = 0; i < j+block_size; i++) {
	for(int b=0; b<block_size; b++) {
	  if(abs(alpha[i*block_size + b]) > mat_norm) mat_norm = abs(alpha[i*block_size + b]);
	  //if(abs(eigenSolverTD.eigenvalues()[i]) > mat_norm) mat_norm = abs(eigenSolverTD.eigenvalues()[i]);	
	}
      }
      //mat_norm = eigenSolverTD.eigenvalues().lpNorm<Eigen::Infinity>();
      printf("mat_norm = %e\n", mat_norm);
      

      
      for(int i=0; i<blocks; i++) {
	for(int b=0; b<block_size; b++) {
	  idx = i*block_data_length + b*(block_size + 1);
	  residua[i*block_size + b] = abs(beta[idx] * eigenSolverTD.eigenvectors().col(i*block_size + b)[j+block_size - 1]);
	}
      }
      
      //Halting check
      if (nEv <= j+block_size) {
	num_converged = 0;
	for(int i=0; i<nEv; i++) {
	  if(residua[i] < tol * mat_norm) num_converged++;
	}
	
	printf("%04d converged eigenvalues at iter %d\n", num_converged, j);
	
	if (num_converged >= nEv) convergence = true;
	
      }
    }
    printf("iter %d complete\n", j);
    j += block_size;
  }
  
  double t2l = clock() - t1;
  
  // Post computation report  
  if (!convergence) {    
    printf("lanczos failed to compute the requested %d vectors with a %d Krylov space\n", nEv, nKr);
  } else {
    printf("lanczos computed the requested %d vectors in %d steps in %e secs.\n", nEv, j, t2l/CLOCKS_PER_SEC);
    
    // Compute eigenvalues
    rotateVecsComplex(kSpace, eigenSolverTD.eigenvectors(), 0, j, j);
    computeEvals(mat, kSpace, residua, evals, nEv);
    for (int i = 0; i < nEv; i++) {
      printf("EigValue[%04d]: (%+.16e, %+.16e) residual %.16e\n", i, evals[i].real(), evals[i].imag(), residua[i]);
    }
    
    for (int i = 0; i < nEv; i++) {
      printf("EigenComp[%04d]: %+.16e\n", i, (evals[i].real() - eigenSolverRef.eigenvalues()[i])/eigenSolverRef.eigenvalues()[i]); 
    }
  }
}
