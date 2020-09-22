#pragma once

#include "linAlgHelpers.h"

std::vector<double> ritz_mat;
std::vector<Complex> block_ritz_mat;

//Functions used in the lanczos algorithm
//---------------------------------------

void iterRefineReal(std::vector<Complex*> &kSpace, std::vector<Complex*> &r, std::vector<double> &alpha, std::vector<double> &beta, int j) {
  
  std::vector<Complex> s(j+1);
  measOrthoDev(kSpace, r, s, j);
  double err = 0.0;
  for(int i=0; i<j+1; i++) err = std::max(err, abs(s[i].real()));
  double cond = (DBL_EPSILON)*beta[j];
  
  int count = 0;
  while (count < 1 && err > cond ) {
    
    // r = r - s_{i} * v_{i}
    orthogonalise(r[0], kSpace, j);
    //alpha[j] += s[j].real();
    //beta[j-1] += s[j-1].real();    
    count++;
    
    measOrthoDev(kSpace, r, s, j);
    err = 0.0;
    for(int i=0; i<j+1; i++) err = std::max(err, abs(s[i].real()));
    cond = (DBL_EPSILON)*beta[j];
    
  }   
}

// Block caxpy r[k] = r[k] + s_{k,j} * vecs_{j} 
void CAXPY(std::vector<Complex*> vecs, std::vector<Complex*> &r, std::vector<Complex> &s, int j, bool plus) {

  int block_size = (int)r.size();
  int idx = 0;
  for(int i=0; i<j+block_size; i++) {
    for(int k=0; k<block_size; k++) {
      idx = i*block_size + k; 
      if(plus) caxpy(+s[idx], vecs[i], r[k]);
      else     caxpy(-s[idx], vecs[i], r[k]);
    }
  }
}


void iterRefineBlock(std::vector<Complex*> &kSpace, std::vector<Complex*> &r, std::vector<Complex> &alpha, std::vector<Complex> &beta, int j) {

  int block_size = (int)r.size();
  int alpha_block_offset = j * block_size;
  int beta_block_offset  = (j - block_size) * block_size;
  
  // r = r - s_{i} * v_{i}
  std::vector<Complex> s((j + block_size) * block_size, 0.0);
  
  int count = 0;
  while (count < 2) {    
    measOrthoDev(kSpace, r, s, j);    
    // r = r - s_{i} * v_{i}
    CAXPY(kSpace, r, s, j, false);
    // Update alpha and beta blocks
    int idx = 0;
    for(int i=0; i<block_size; i++) {
      for(int k=0; k<block_size; k++) {
	idx = i*block_size + k;
	//alpha[alpha_block_offset + idx] += s[alpha_block_offset + idx];
	//if(j>0) beta[beta_block_offset   + idx] += s[beta_block_offset  + idx];
      }
    }    
    count++;
  }
}


//The Engine
void lanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		 std::vector<double> &beta, std::vector<double> &alpha,
		 std::vector<Complex*> &r, int num_keep, int j,
		 double a_min, double a_max, int poly_deg) {
  
  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  if(a_min == 0.0 || a_max == 0.0) matVec(mat, r[0], kSpace[j]);
  else chebyOp(mat, r[0], kSpace[j], a_min, a_max, poly_deg);

  //a_j = v_j^dag * r
  alpha[j] = (cDotProd(kSpace[j], r[0])).real();    

  //r = r - a_j * v_j
  axpy(-alpha[j], kSpace[j], r[0]);

  int start = (j > num_keep && j>0) ? j - 1 : 0;
  for (int i = start; i < j; i++) {

    // r = r - b_{j-1} * v_{j-1}
    axpy(-beta[i], kSpace[i], r[0]);
  }

  // Orthogonalise r against the K space
  //if (j > 0) for (int k = 0; k < 10; k++) orthogonalise(r[0], kSpace, j);
  if (j > 0) for (int k = 0; k < 1; k++) iterRefineReal(kSpace, r, alpha, beta, j);
  
  //b_j = ||r|| 
  beta[j] = normalise(r[0]);

  //Prepare next step.
  copy(kSpace[j+1], r[0]);
}

void arnoldiStep(Complex **mat, std::vector<Complex*> &kSpace,
		 std::vector<Complex*> &upperHess,
		 std::vector<Complex*> &r, int j) {
  
  matVec(mat, r[0], kSpace[j]);

  double norm_pre = norm(r[0]);
  for (int i = 0; i < j+1; i++) {
    //H_{j,i}_j = v_i^dag * r
    upperHess[i][j] = cDotProd(kSpace[i], r[0]);
    //printf("upperHess[%d][%d] = (%e,%e)\n", i, j, upperHess[i][j].real(), upperHess[i][j].imag());
    //r = r - H_{j,i} * v_j 
    caxpy(-1.0*upperHess[i][j], kSpace[i], r[0]);
  }
  
  double norm_post = norm(r[0]);
  upperHess[j+1][j].real(norm_post);
  //printf("Residual norm = %e\n", norm_post); 

  if(norm_post < 1e-12 || norm_pre < 1e-12){
    printf("Congratulations! You have reached an invariant subspace at iter %d, beta_pre = %e, beta_post = %e\n", j, norm_pre, norm_post);
    //exit(0);
  }
  
  // Orthogonalise r against the K space
  if(norm_post < 0.717*norm_pre && j > 0) {
    
    // reorthogonalise r against the K space
    //printf("beta = %e < %e: Reorthogonalise at step %d\n",
    //norm_post, 0.717*norm_pre, j);
    
    Complex alpha = 0.0;
    for(int i=0; i < j+1; i++) {
      alpha = cDotProd(kSpace[i], r[0]);
      upperHess[i][j] += alpha;
      caxpy(-1.0*alpha, kSpace[i], r[0]);
    }
    upperHess[j+1][j].real(norm(r[0]));
  }
  
  //Prepare next step.
  normalise(r[0]);
  copy(kSpace[j+1], r[0]);
}

#if 1
void arnoldiStep2(Complex **mat, std::vector<Complex*> &kSpace,
		  std::vector<Complex*> &upperHess,
		  std::vector<Complex*> &r, double &beta, int j) {

  beta = norm(r[0]);
  if (beta < 1e-15) {
    printf("Invariant subspace at iter %d\n", j);
    exit(0);
  }

  upperHess[j+1][j].real(beta);
  upperHess[j+1][j].imag(0.0);
    
  zero(kSpace[j]);
  caxpy(1/beta, r[0], kSpace[j]);    
  matVec(mat, r[0], kSpace[j]);

  double nrm = norm(r[0]);
  
  for (int i = 0; i < j+1; i++) {
    //H_{j,i}_j = v_i^dag * r
    upperHess[i][j] = cDotProd(kSpace[i], r[0]);
    //printf("upperHess[%d][%d] = (%e,%e)\n", i, j, upperHess[i][j].real(), upperHess[i][j].imag());
    //r = r - H_{j,i} * v_j 
    caxpy(-1.0*upperHess[i][j], kSpace[i], r[0]);
  }
  
  beta = norm(r[0]);
  
  if(beta < 0.717*nrm && j > 0) {
    // reorthogonalise r against the K space
    printf("beta = %e < %e: Reorthogonalise at step %d\n", beta, 0.717*nrm, j);
    Complex alpha = 0.0;
    for(int i=0; i < j+1; i++) {
      alpha = cDotProd(kSpace[i], r[0]);
      upperHess[i][j] += alpha;
      caxpy(-1.0*alpha, kSpace[i], r[0]);
    }
    beta = norm(r[0]);
  }
}
#endif


void reorder(std::vector<Complex*> &kSpace, std::vector<double> alpha, int nKr, bool reverse) {
  int i = 0;
  Complex temp[Nvec];
  if (reverse) {
    while (i < nKr) {
      if ((i == 0) || (alpha[i - 1] >= alpha[i]))
	i++;
      else {
	double tmp = alpha[i];
	alpha[i] = alpha[i - 1];
	alpha[--i] = tmp;
	copy(temp, kSpace[i]);
	copy(kSpace[i], kSpace[i-1]);
	copy(kSpace[i-1], temp);
      }
    }
  } else {
    while (i < nKr) {
      if ((i == 0) || (alpha[i - 1] <= alpha[i])) 
	i++;
      else {
	double tmp = alpha[i];
	alpha[i] = alpha[i - 1];
	alpha[--i] = tmp;
	copy(temp, kSpace[i]);
	copy(kSpace[i], kSpace[i-1]);
	copy(kSpace[i-1], temp);
      }
    }
  }
}

void blockLanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		      std::vector<Complex> &beta, std::vector<Complex> &alpha,
		      std::vector<Complex*> &r, int num_keep, int j, int block_size,
		      double a_min, double a_max, int poly_deg) {

  int block_offset = j * block_size;
  int block_data_length = block_size * block_size;
  
  //Compute r = A * v_j - b_{j-i} * v_{j-1}      
  //r = A * v_j
  for(int b=0; b<block_size; b++) {
    if(a_min == 0.0 || a_max == 0.0) matVec(mat, r[b], kSpace[j+b]);
    else chebyOp(mat, r[b], kSpace[j+b], a_min, a_max, poly_deg);
  }

  //a_j = v_j^dag * r
  int idx = 0;
  for(int b=0; b<block_size; b++) {
    for(int c=0; c<block_size; c++) {
      idx = b*block_size + c;
      alpha[block_offset + idx] = cDotProd(kSpace[j+b], r[c]);    
    }
  }
  //r = r - a_j * v_j CAXPY this
  for(int b=0; b<block_size; b++) {
    for(int c=0; c<block_size; c++) {
      idx = b*block_size + c;
      caxpy(-alpha[block_offset + idx], kSpace[j+b], r[c]);
    }
  }

  // r = r - b_{j-1} * v_{j-1}
  int start = (j > num_keep) ? j - block_size : 0;
  for (int i = start/block_size; i < j/block_size; i++) {
    int offset = i*block_data_length;
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = b*block_size + c;
	caxpy(-beta[offset + idx], kSpace[j - block_size + b], r[c]);
      }
    }
  }
  
  // Orthogonalise r against the kSpace
  //if(j>0) for(int b=0; b<block_size; b++) orthogonalise(r[b], kSpace, j);
  iterRefineBlock(kSpace, r, alpha, beta, j);

  gramSchmidtRecursive(r, beta, block_offset, 10);

  for(int b=0; b<block_size; b++) {
    for(int c=0; c<b+1; c++) {
      idx = b*block_size + c;
      if(abs(beta[block_offset + idx]) < 1e-10)
	printf("WARNING: |beta(%d,%d)| = %e < 1e-10\n",
	       c, b, abs(beta[block_offset + idx]));
    }
  }
  
  //Prepare next step.
  for(int b=0; b<block_size; b++) copy(kSpace[j+block_size + b], r[b]);
}



double eigensolveFromUpperHess(std::vector<Complex*> &upperHess, Eigen::ComplexEigenSolver<MatrixXcd> &eigenSolverUH, int nKr, int num_locked)
{
  int dim = nKr - num_locked;
  //Construct the Upper Hessenberg matrix H_k      
  MatrixXcd upperHessEigen = MatrixXcd::Zero(dim, dim);      
  for(int k=num_locked; k<nKr; k++) {
    for(int i=num_locked; i<nKr; i++) {
      upperHessEigen(k,i) = upperHess[k][i];
    }
  }
  //cout << upperHessEigen << endl;  
  // Eigensolve dense UH
  eigenSolverUH.compute(upperHessEigen);
  //cout << eigenSolverUH.eigenvalues() << endl << endl;
  //for(int i=0; i<dim; i++) cout << i << " " << abs(eigenSolverUH.eigenvalues()[i]) << endl;
  return upperHessEigen.norm();
}

void qriteration(MatrixXcd &Rmat, MatrixXcd &Qmat, int dim)
{  
  Complex T11, T12, T21, T22, temp, temp2, temp3, U1, U2;
  double dV;

  double tol = 1e-15;
  
  // Allocate the rotation matrices.
  Complex *R11 = (Complex*) malloc((dim-1)*sizeof(Complex));
  Complex *R12 = (Complex*) malloc((dim-1)*sizeof(Complex));
  Complex *R21 = (Complex*) malloc((dim-1)*sizeof(Complex));
  Complex *R22 = (Complex*) malloc((dim-1)*sizeof(Complex));

  // First pass, determine the matrices and do H -> R.
  for(int i = 0; i < dim-1; i++) {
    if (abs(Rmat(i+1, i)) < tol) {
      R11[i] = R12[i] = R21[i] = R22[i] = 0;
      Rmat(i+1, i) = 0;
      continue;
    }
    
    dV = sqrt(norm(Rmat(i, i)) + norm(Rmat(i+1, i)));
    U1 = Rmat(i, i);
    dV = (U1.real() > 0) ? dV : -dV;
    U1 += dV;
    U2 = Rmat(i+1, i);
        
    T11 = conj(U1);
    T11 /= dV;
    R11[i] = conj(T11);

    T12 = conj(U2);
    T12 /= dV;
    R12[i] = conj(T12);
    
    T21 = conj(T12);
    temp = conj(U1);
    temp /= U1;
    T21 *= temp;
    R21[i] = conj(T21);

    temp = U2 / U1;
    T22 = T12 * temp;
    R22[i] = conj(T22);

    // Do the H_kk and set the H_k+1k to zero
    temp = Rmat(i, i);
    temp2 = T11 * temp;
    temp3 = T12 * Rmat(i+1, i);
    temp2 += temp3;
    Rmat(i, i) -= temp2;
    Rmat(i+1, i) = 0;
    // Continue for the other columns
    for(int j=i+1; j < dim; j++) {
      temp = Rmat(i, j);
      temp2 = T11 * temp;
      temp2 += T12 * Rmat(i+1, j);
      Rmat(i, j) -= temp2;
      
      temp2 = T21 * temp;
      temp2 += T22 * Rmat(i+1, j);
      Rmat(i+1, j) -= temp2;
    }
  }

  // Rotate R and V, i.e. H->RQ. V->VQ 
  for(int j = 0; j < dim - 1; j++) {
    if(abs(R11[j]) > tol) {
      for(int i = 0; i < j+2; i++) {
	temp = Rmat(i, j);
	temp2 = R11[j] * temp;
	temp2 += R12[j] * Rmat(i, j+1);
	Rmat(i, j) -= temp2;
	
	temp2 = R21[j] * temp;
	temp2 += R22[j] * Rmat(i, j+1);
	Rmat(i, j+1) -= temp2;
      }
      
      for(int i = 0; i < dim; i++) {
	temp = Qmat(i, j);
	temp2 = R11[j] * temp;
	temp2 += R12[j] * Qmat(i, j+1);
	Qmat(i, j) -= temp2;
	
	temp2 = R21[j] * temp;
	temp2 += R22[j] * Qmat(i, j+1);
	Qmat(i, j+1) -= temp2;
      }
    }
  }
  
  // Free allocated vectors
  free(R11);
  free(R12);
  free(R21);
  free(R22);
} 

int qrFromUpperHess(MatrixXcd &upperHess, MatrixXcd &Qmat, MatrixXcd &Rmat, int nKr, int num_locked)
{  
  int dim = nKr - num_locked;
  for(int k=0; k<dim; k++) {
    for(int i=0; i<dim; i++) {
      Rmat(k,i) = upperHess(k,i);
    }
  }

  Eigen::ComplexSchur<MatrixXcd> schurUH;
  schurUH.compute(Rmat);  
  //cout << schurUH.matrixT() << endl << endl;
  //cout << schurUH.matrixU() << endl << endl;
  //cout << Rmat << endl << endl;
  //cout << schurUH.matrixU() * schurUH.matrixT() * schurUH.matrixU().adjoint() << endl << endl;
  
  double tol = 1e-15;
  Complex temp, disc, dist1, dist2, eval;
  int iter = 0;
  
  // The convergence is much faster if we start in the lower corner
  // of the matrix
  //for (int k = 0; k < dim-1; k++) {
  for (int k = dim-2; k >= 0; k--) {
    while (iter < 10000) {
      if(abs(Rmat(k+1, k)) < tol) {
	Rmat(k+1, k) = 0;
	break;
      }
      
      // Calculate the eigenvalues of the 2x2 matrix
      temp = Rmat(k, k) - Rmat(k+1, k+1);
      temp *= temp;
      temp /= 4;
      
      disc = Rmat(k+1, k) * Rmat(k, k+1);
      disc += temp;
      disc = sqrt(disc);
      temp = Rmat(k, k) + Rmat(k+1, k+1);
      temp /= 2;
      dist1 = temp + disc;
      dist1 = dist1 - Rmat(k+1, k+1);
      dist2 = temp - disc;
      dist2 = dist2 - Rmat(k+1, k+1);
      if (norm(dist1) < norm(dist2))
	eval = dist1 + Rmat(k+1, k+1);
      else
	eval = dist2 + Rmat(k+1, k+1);
      
      // Shift H with the eigenvalue
      for(int i = 0; i < dim; i++) {
	Rmat(i, i) -= eval;
      }
      
      // Do the QR iteration
      qriteration(Rmat, Qmat, dim);
      
      // Shift H back
      for(int i = 0; i < dim; i++)
	Rmat(i, i) += eval;
      
      iter++;    
    } 
  }
  
  printf("eigensystem iterations = %d\n", iter);

  //cout << Rmat << endl << endl;
  //cout << Qmat << endl << endl;
  //cout << Qmat * Rmat << endl << endl;
  
  //for(int i=0; i<dim; i++) cout << Rmat(i,i) << endl;
  
  return iter;  
}


void eigensolveFromArrowMat(int num_locked, int arrow_pos, int nKr, std::vector<double> &alpha, std::vector<double> &beta, std::vector<double> &residua, bool reverse) {
  
  int dim = nKr - num_locked;
  
  // Eigen objects
  MatrixXd A = MatrixXd::Zero(dim, dim);
  ritz_mat.resize(dim * dim);
  for (int i = 0; i < dim * dim; i++) ritz_mat[i] = 0.0;

  // Optionally invert the spectrum
  if (reverse) {
    for (int i = num_locked; i < nKr - 1; i++) {
      alpha[i] *= -1.0;
      beta[i] *= -1.0;
    }
    alpha[nKr - 1] *= -1.0;
  }
  
  // Construct arrow mat A_{dim,dim}
  for (int i = 0; i < dim; i++) {    
    // alpha populates the diagonal
    A(i,i) = alpha[i + num_locked];
  }
  
  for (int i = 0; i < arrow_pos - 1; i++) {  
    // beta populates the arrow
    A(i, arrow_pos - 1) = beta[i + num_locked];
    A(arrow_pos - 1, i) = beta[i + num_locked];
  }
  
  for (int i = arrow_pos - 1; i < dim - 1; i++) {
    // beta populates the sub-diagonal
    A(i, i + 1) = beta[i + num_locked];
    A(i + 1, i) = beta[i + num_locked];
  }
  
  // Eigensolve the arrow matrix 
  Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver;
  eigenSolver.compute(A);
  
  // repopulate ritz matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      //Place data in COLUMN major 
      ritz_mat[dim * i + j] = eigenSolver.eigenvectors().col(i)[j];
      //printf("%+.4e ",ritz_mat[dim * i + j]);      
    }
    //printf("\n");
  }
  
  for (int i = 0; i < dim; i++) {
    residua[i + num_locked] = fabs(beta[nKr - 1] * eigenSolver.eigenvectors().col(i)[dim - 1]);
    // Update the alpha array
    alpha[i + num_locked] = eigenSolver.eigenvalues()[i];
    //printf("EFAM: resid = %e, alpha = %e\n", residua[i + num_locked], alpha[i + num_locked]);
  }

  // Put spectrum back in order
  if (reverse) {
    for (int i = num_locked; i < nKr; i++) { alpha[i] *= -1.0; }
  }  
}

void eigensolveFromBlockArrowMat(int num_locked, int arrow_pos, int nKr, int block_size, int restart_iter, std::vector<Complex> &alpha, std::vector<Complex> &beta, std::vector<Complex> &arrow_eigs, std::vector<double> &residua, bool reverse) {

  int block_data_length = block_size * block_size;
  int dim = nKr - num_locked;
  if (dim % block_size != 0) {
    printf("dim = %d modulo block_size = %d != 0", dim, block_size);
    exit(0);
  }  
  int blocks = dim / block_size;
  
  if (arrow_pos % block_size != 0) {
    printf("arrow_pos = %d modulo block_size = %d != 0", arrow_pos, block_size);
    exit(0);
  }
  
  int block_arrow_pos = arrow_pos / block_size;
  int num_locked_offset = (num_locked / block_size) * block_data_length;  

  // Eigen objects
  MatrixXcd T = MatrixXcd::Zero(dim, dim);
  block_ritz_mat.resize(dim * dim);
  int idx = 0;
  
  // Populate the r and eblocks
  for (int i = 0; i < block_arrow_pos; i++) {
    for (int b = 0; b < block_size; b++) {
      
      // E block
      idx = i * block_size + b;
      T(idx, idx) = arrow_eigs[idx + num_locked];
      
      for (int c = 0; c < block_size; c++) {
	// r blocks
	idx = num_locked_offset + b * block_size + c;
	T(arrow_pos + c, i * block_size + b) = beta[i * block_data_length + idx];
	T(i * block_size + b, arrow_pos + c) = conj(beta[i * block_data_length + idx]);
      }
    }
  }

  // Add the alpha blocks
  for (int i = block_arrow_pos; i < blocks; i++) {
    for (int b = 0; b < block_size; b++) {
      for (int c = 0; c < block_size; c++) {
	idx = num_locked_offset + b * block_size + c;
	T(i * block_size + b, i * block_size + c) = alpha[i * block_data_length + idx];
      }
    }
  }

  // Add the beta blocks
  for (int i = block_arrow_pos; i < blocks - 1; i++) {
    for (int b = 0; b < block_size; b++) {
      for (int c = 0; c < b + 1; c++) {
	idx = num_locked_offset + b * block_size + c;
	// Sub diag
	T((i + 1) * block_size + c, i * block_size + b) = beta[i * block_data_length + idx];
	// Super diag
	T(i * block_size + b, (i + 1) * block_size + c) = conj(beta[i * block_data_length + idx]);
      }
    }
  }

  // Invert the spectrum due to Chebyshev (except the arrow diagonal)
  if (reverse) {
    for (int b = 0; b < dim; b++) {
      for (int c = 0; c < dim; c++) {
	T(c, b) *= -1.0;
	if (restart_iter > 0)
	  if (b == c && b < arrow_pos && c < arrow_pos) T(c, b) *= -1.0;
      }
    }
  }

  // Eigensolve the arrow matrix
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eigensolver;
  eigensolver.compute(T);

  // Populate the aroow_eigs array with eigenvalues
  for (int i = 0; i < dim; i++) arrow_eigs[i + num_locked] = eigensolver.eigenvalues()[i];
  
  // Repopulate ritz matrix: COLUMN major
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++) block_ritz_mat[dim * i + j] = eigensolver.eigenvectors().col(i)[j];
  
  for (int i = 0; i < blocks; i++) {
    for (int b = 0; b < block_size; b++) {
      idx = b * (block_size + 1);
      residua[i * block_size + b + num_locked] = abs(beta[nKr * block_size - block_data_length + idx] * block_ritz_mat[dim * (i * block_size + b + 1) - 1]);
    }
  }  
}

void computeEvals(Complex **mat, std::vector<Complex*> &kSpace, std::vector<double> &residua, std::vector<Complex> &evals, int nEv) {
  
  //temp vector
  Complex temp[Nvec];
  for (int i = 0; i < nEv; i++) {
    // r = A * v_i
    matVec(mat, temp, kSpace[i]);

    // lambda_i = v_i^dag A v_i / (v_i^dag * v_i)
    //cout << i << " " << cDotProd(kSpace[i], temp) << " " << norm(kSpace[i]) << " " << norm(temp) << " " << cDotProd(kSpace[i], temp) / norm(kSpace[i]) << endl;
    
    //evals[i] = cDotProd(kSpace[i], temp) / norm(kSpace[i]);
    evals[i] = cDotProd(kSpace[i], temp);

    for(int j=0; j<3; j++) {
      //cout << "elem " << j << ": " << evals[i] * kSpace[i][j] << " " << temp[j] << " " << temp[j]/(evals[i] * kSpace[i][j]) << endl;      
    }
    
    // Measure ||lambda_i*v_i - A*v_i||
    Complex n_unit(-1.0, 0.0);
    caxpby(evals[i], kSpace[i], n_unit, temp);
    residua[i] = norm(temp);
  }
}

void rotateVecs(std::vector<Complex*> &vecs, Eigen::MatrixXd mat, int num_locked, int iter_keep, int dim) {

  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {    
    
    //put jth row of V_k in temp
    Complex tmp[dim];  
    for(int i=0; i<dim; i++) {
      tmp[i] = vecs[i+num_locked][j];      
    }

    //take product of jth row of V_k and ith column of mat (ith eigenvector of T_k) 
    Complex sum = 0.0;
    for(int i=0; i<iter_keep; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<dim; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }
      
      //Update the Ritz vector
      vecs[i+num_locked][j] = sum;
      sum = 0.0;
    }
  }
}

void rotateVecsComplex(std::vector<Complex*> &vecs, Eigen::MatrixXcd mat, int num_locked, int iter_keep, int dim) {

  //loop over rows of V_k
  for(int j=0; j<Nvec; j++) {    
    
    //put jth row of V_k in temp
    Complex tmp[dim];  
    for(int i=0; i<dim; i++) {
      tmp[i] = vecs[i+num_locked][j];      
    }

    //take product of jth row of V_k and ith column of mat (ith eigenvector of T_k) 
    Complex sum = 0.0;
    for(int i=0; i<iter_keep; i++) {
      
      //Loop over elements to get the y_i[j]th element 
      for(int l=0; l<dim; l++) {
	sum += tmp[l]*mat.col(i)[l];
      }
      
      //Update the Ritz vector
      vecs[i+num_locked][j] = sum;
      sum = 0.0;
    }
  }
}


void computeKeptRitz(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, std::vector<double> &beta) {
  
  int dim = nKr - num_locked;
  MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = ritz_mat[j*dim + i];  
  rotateVecs(kSpace, mat, num_locked, iter_keep, dim); 
  
  //Update beta and residual
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * mat.col(i)[nKr-num_locked-1];  
}

void computeKeptRitzComplex(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, int block_size, std::vector<Complex> &beta) {
  
  int dim = nKr - num_locked;
  MatrixXcd mat = MatrixXcd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = block_ritz_mat[j*dim + i];  
  rotateVecsComplex(kSpace, mat, num_locked, iter_keep, dim); 
  
  //Update beta and residua
  for(int b=0; b<block_size; b++) copy(kSpace[num_locked + iter_keep + b], kSpace[nKr + b]);

  // Compute new r blocks
  // Use Eigen, it's neater
  
  MatrixXcd beta_mat = MatrixXcd::Zero(block_size, block_size);
  MatrixXcd ri = MatrixXcd::Zero(block_size, block_size);
  MatrixXcd ritzi = MatrixXcd::Zero(block_size, block_size);
  int blocks = iter_keep / block_size;
  int idx = 0;
  int block_data_length = block_size * block_size;
  int beta_offset = nKr * block_size - block_data_length;
  int num_locked_offset = num_locked * block_size;

  for (int b = 0; b < block_size; b++) {
    for (int c = 0; c < b + 1; c++) {
      idx = b * block_size + c;
      beta_mat(c, b) = beta[beta_offset + idx];
    }
  }
  for (int i = 0; i < blocks; i++) {
    for (int b = 0; b < block_size; b++) {
      for (int c = 0; c < block_size; c++) {
	idx = i * block_size * dim + b * dim + (dim - block_size) + c;
	ritzi(c, b) = block_ritz_mat[idx];
      }
    }
    
    ri = beta_mat * ritzi;
    for (int b = 0; b < block_size; b++) {
      for (int c = 0; c < block_size; c++) {
	idx = num_locked_offset + b * block_size + c;
	beta[i * block_data_length + idx] = ri(c, b);
      }
    }
  }
}


void permuteVecs(std::vector<Complex*> &kSpace, Eigen::MatrixXd mat, int num_locked, int size){

  std::vector<int> pivots(size);
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      if(mat(i,j) == 1) {
	pivots[j] = i;
      }
    }
  }

  // Identify cycles in the permutation array.
  // We shall use the sign bit as a marker. If the
  // sign is negative, the vectors has already been
  // swapped into the correct place. A positive
  // value indicates the start of a new cycle.

  Complex temp[Nvec];
  for (int i=0; i<size; i++) {
    //Cycles always start at 0, hence OR statement
    if(pivots[i] > 0 || i==0) {
      int k = i;
      // Identify vector to be placed at i
      int j = pivots[i];
      pivots[i] = -pivots[i];
      while (j > i) {
	copy(temp, kSpace[k+num_locked]);
	copy(kSpace[k+num_locked], kSpace[j+num_locked]);
	copy(kSpace[j+num_locked], temp);
	pivots[j] = -pivots[j];
	k = j;
	j = -pivots[j];
      }
    } else {
      //printf("%d already swapped\n", i);
    }
  }
  for (int i=0; i<size; i++) {
    if (pivots[i] > 0) {
      printf("Error at %d\n", i);
      exit(0);
    }
  }
}

void computeKeptRitzLU(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, int batch, std::vector<double> &beta, int iter) {
  
  int offset = nKr + 1;
  int dim = nKr - num_locked;

  printf("dim = %d\n", dim);
  printf("iter_keep = %d\n", iter_keep);
  printf("num_locked = %d\n", num_locked);
  printf("kspace size = %d\n", (int)kSpace.size());

  int batch_size = batch;
  int full_batches = iter_keep/batch_size;
  int batch_size_r = iter_keep%batch_size;
  bool do_batch_remainder = (batch_size_r != 0 ? true : false);
  
  printf("batch_size = %d\n", batch_size);
  printf("full_batches = %d\n", full_batches);
  printf("batch_size_r = %d\n", batch_size_r);
  
  if ((int)kSpace.size() < offset + batch_size) {
    for (int i = kSpace.size(); i < offset + batch_size; i++) {
      kSpace.push_back(new Complex[Nvec]);
    }
  }

  // Zero out the work space
  for (int i = offset; i < offset + batch_size; i++) {
    zero(kSpace[i]);
  }

  MatrixXd mat = MatrixXd::Zero(dim, iter_keep);
  for (int j = 0; j < iter_keep; j++) 
    for (int i = 0; i < dim; i++) 
      mat(i,j) = ritz_mat[j*dim + i];
  
  Eigen::FullPivLU<MatrixXd> matLU(mat);
  // RitzLU now contains the LU decomposition
  
  MatrixXd matUpper = MatrixXd::Zero(dim,iter_keep);
  matUpper = matLU.matrixLU().triangularView<Eigen::Upper>();
  MatrixXd matLower = MatrixXd::Identity(dim,dim);
  matLower.block(0,0,dim,iter_keep).triangularView<Eigen::StrictlyLower>() = matLU.matrixLU();

  permuteVecs(kSpace, matLU.permutationP().inverse(), num_locked, dim);    
  
  // Defines the column element (row index)
  // from which we reference other indicies
  int i_start, i_end, j_start, j_end;
  
  // Do L Portion
  //---------------------------------------------------------------------------
  // Loop over full batches
  for (int b = 0; b < full_batches; b++) {

    // batch triangle
    i_start = b*batch_size;
    i_end   = (b+1)*batch_size; 
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = j; i < i_end; i++) {	
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    //batch pencil
    i_start = (b+1)*batch_size;
    i_end   = dim;
    j_start = b*batch_size;
    j_end   = (b+1)*batch_size;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }
  
  if(do_batch_remainder) {
    // remainder triangle
    i_start = full_batches*batch_size;
    i_end   = iter_keep;
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = j; i < i_end; i++) {
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }   
    }
    //remainder pencil
    i_start = iter_keep;
    i_end   = dim;
    j_start = full_batches*batch_size;
    j_end   = iter_keep;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {	
	axpy(matLower.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }
  }
    
  // Do U Portion
  //---------------------------------------------------------------------------
  if(do_batch_remainder) {

    // remainder triangle
    i_start = full_batches*batch_size;
    i_end   = iter_keep;
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < j+1; i++) {	
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    //remainder pencil
    i_start = 0;
    i_end   = full_batches*batch_size; 
    j_start = full_batches*batch_size;
    j_end   = iter_keep;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < i_end; i++) {
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }
  
  // Loop over full batches
  for (int b = full_batches-1; b >= 0; b--) {

    // batch triangle
    i_start = b*batch_size;
    i_end   = (b+1)*batch_size; 
    j_start = i_start;
    j_end   = i_end;
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      for (int i = i_start; i < j+1; i++) {	
	axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
      }
    }

    if(b>0) {
      //batch pencil
      i_start = 0;
      i_end   = b*batch_size; 
      j_start = b*batch_size;
      j_end   = (b+1)*batch_size;
      for (int j = j_start; j < j_end; j++) {
	int k = offset + j - j_start;
	for (int i = i_start; i < i_end; i++) {	  
	  axpy(matUpper.col(j)[i], kSpace[num_locked + i], kSpace[k]);
	}
      }
    }
    // copy back to correct position
    for (int j = j_start; j < j_end; j++) {
      int k = offset + j - j_start;
      copy(kSpace[j + num_locked], kSpace[k]);
      zero(kSpace[k]);
    }    
  }

  permuteVecs(kSpace, matLU.permutationQ().inverse(), num_locked, iter_keep);

  //Update residual
  copy(kSpace[num_locked + iter_keep], kSpace[nKr]);
  
  //Update beta
  for (int i = 0; i < iter_keep; i++)
    beta[i + num_locked] = beta[nKr - 1] * ritz_mat[dim * (i + 1) - 1];
  
}
