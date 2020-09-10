#pragma once

#include <omp.h>

//Simple Complex Linear Algebra Helpers
void zero(Complex *x) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] = 0.0;
}

void copy(Complex *x, Complex *y) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] = y[i];
}

void ax(double a, Complex *x) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] *= a;
}

void cax(Complex a, Complex *x) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) x[i] *= a;
}

void axpy(double a, Complex *x, Complex *y) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) y[i] += a*x[i];
}

void caxpy(Complex a, Complex *x, Complex *y) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) y[i] += a*x[i];
}

void axpby(double a, Complex *x, double b, Complex *y) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) {
    y[i] *= b;
    y[i] += a*x[i];
  }
}

void caxpby(Complex a, Complex *x, Complex b, Complex *y) {
#pragma omp parallel for
  for(int i=0; i<Nvec; i++) {
    y[i] *= b;
    y[i] += a*x[i];
  }
}

Complex dotProd(Complex *x, Complex *y) {
  Complex prod = 0.0;
#pragma omp parallel for reduction(+:prod) 
  for(int i=0; i<Nvec; i++) prod += x[i]*y[i];
  return prod;
}

Complex cDotProd(const Complex *x, const Complex *y) {
  Complex prod = 0.0;
#pragma omp parallel for reduction(+:prod) 
  for(int i=0; i<Nvec; i++) prod += conj(x[i])*y[i];
  return prod;
}

double norm2(Complex *x) {
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for(int i=0; i<Nvec; i++) sum += (conj(x[i])*x[i]).real();
  return sum;
}

double norm(Complex *x) {
  return sqrt(norm2(x));
}

double normalise(Complex *x) {
  double sum = norm(x);
  ax(1.0/sum, x); 
  return sum;
}

//Orthogonalise r against the j vectors in vectorSpace
void orthogonalise(Complex *r, std::vector<Complex*> vectorSpace, int j) {
  
  Complex s = 0.0;
  for(int i=0; i<j; i++) {
    s = cDotProd(vectorSpace[i], r);
    caxpy(-s, vectorSpace[i], r);
  }
}

void matVec(Complex **mat, Complex *out, Complex *in) {
  
  Complex temp[Nvec];
  zero(temp);
  //Loop over rows of matrix
  //#pragma omp parallel for 
  for(int i=0; i<Nvec; i++) {
    temp[i] = dotProd(&mat[i][0], in);    
  }
  copy(out, temp);  
}

void chebyOp(Complex **mat, Complex *out, Complex *in, double a, double b, int poly_deg) {
  
  // Compute the polynomial accelerated operator.
  double delta = (b - a) / 2.0;
  double theta = (b + a) / 2.0;
  double sigma1 = -delta / theta;
  double sigma;
  double d1 = sigma1 / delta;
  double d2 = 1.0;
  double d3;

  // out = d2 * in + d1 * out
  // C_1(x) = x
  matVec(mat, out, in);
  caxpby(d2, in, d1, out);
  
  Complex tmp1[Nvec];
  Complex tmp2[Nvec];
  Complex tmp3[Nvec];
  
  copy(tmp1, in);
  copy(tmp2, out);
  
  // Using Chebyshev polynomial recursion relation,
  // C_{m+1}(x) = 2*x*C_{m} - C_{m-1}
  
  double sigma_old = sigma1;
  
  // construct C_{m+1}(x)
  for (int i = 2; i < poly_deg; i++) {
    sigma = 1.0 / (2.0 / sigma1 - sigma_old);
      
    d1 = 2.0 * sigma / delta;
    d2 = -d1 * theta;
    d3 = -sigma * sigma_old;

    // mat*C_{m}(x)
    matVec(mat, out, tmp2);

    Complex d1c(d1, 0.0);
    Complex d2c(d2, 0.0);
    Complex d3c(d3, 0.0);

    copy(tmp3, tmp2);
    
    caxpby(d3c, tmp1, d2c, tmp3);
    caxpy(d1c, out, tmp3);
    copy(tmp2, tmp3);
    
    sigma_old = sigma;
  }
  copy(out, tmp2);
}

// Measure the deviation from orthogonality of r[k] with vecs[i],
// place results in s[i*block_size + k]
void measOrthoDev(std::vector<Complex*> vecs, std::vector<Complex*> r, std::vector<Complex> &s, int j) {

  int block_size = (int)r.size();
  int idx = 0;
  for(int i=0; i<j+block_size; i++) {
    for(int k=0; k<block_size; k++) {
      idx = i*block_size + k;
      s[idx] = cDotProd(vecs[i], r[k]);
    }
  }
}

bool orthoCheck(std::vector<Complex*> vecs, int size, bool verbose = false) {
  
  bool orthed = true;
  const Complex Unit(1.0,0.0);
  
  for(int i=0; i<size; i++) {
    for(int j=0; j<size; j++) {
      Complex cnorm = cDotProd(vecs[j], vecs[i]);
      if(j != i) {
	if(verbose) printf("Norm <%d|%d>^2 = (%e,%e): %e\n", i, j, cnorm.real(), cnorm.imag(), abs(cnorm));
	if(abs(cnorm) > 1e-15) orthed = false;
      } else {
	if(verbose) printf("Norm <%d|%d>^2 = (%e,%e): %e\n", i, j, cnorm.real(), cnorm.imag(), abs(Unit - cnorm));
	if (abs(Unit - cnorm) > 1e-15) orthed = false;
      }
    }
  }
  return orthed;
}

void updateBlockBeta(std::vector<Complex> tmp, std::vector<Complex> &beta, int k, int block_offset, int block_size) {

  if(k == 0) {
    //Copy over the tmp matrix to block beta, Beta = R_0
    int idx = 0;
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = b*block_size + c;
	beta[block_offset + idx] = tmp[idx];
      }
    }
  } else {
    // Compute BetaNew_ac = (R_k)_ab * Beta_bc
    // Use Eigen, it's neater
    MatrixXcd betaNEigen = MatrixXcd::Zero(block_size, block_size);
    MatrixXcd betaEigen  = MatrixXcd::Zero(block_size, block_size);
    MatrixXcd RkEigen    = MatrixXcd::Zero(block_size, block_size);
    int idx = 0;
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = b*block_size + c;
	betaEigen(c,b) = beta[block_offset + idx];
	RkEigen(c,b)   = tmp[idx];
      }
    }

    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = b*block_size + c;
	printf("(%e,%e) ", tmp[idx].real(), tmp[idx].imag()); 
      }
      printf("\n");
    }
    betaNEigen = RkEigen * betaEigen;
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<block_size; c++) {
	idx = b*block_size + c;
	beta[block_offset + idx] = betaNEigen(c,b);
      }
    }
  }
  int idx = 0;
  for(int b=0; b<block_size; b++) {
    for(int c=0; c<block_size; c++) {
      idx = b*block_size + c;
      printf("(%e,%e) ", beta[block_offset + idx].real(), beta[block_offset + idx].imag()); 
    }
    printf("\n");
  }
}


// QR decomposition via modified Gram Scmidt
// NB, QR via modified Gram-Schmidt is numerically unstable.
// A recursive strategy recovers full MP orthonormality
// Q_0 * R_0(V)   -> Q_0 * R_0 = V
// Q_1 * R_1(Q_0) -> Q_1 * R_1 = V * R_0^-1 -> Q_1 * R_1 * R_0 = V
// ...
// Q_k * R_k(Q_{k-1}) -> Q_k * R_k * R_{k-1} * ... * R_0 = V
//
// Where the Q_k are orthonormal to MP and (R_k * R_{k-1} * ... * R_0)^1
// is the matrix that maps V -> Q_k.
void gramSchmidtRecursive(std::vector<Complex*> &vecs, std::vector<Complex> &beta,
			  int block_offset, int k_max) {

  int block_size = (int)vecs.size();
  std::vector<Complex> tmp(block_size * block_size, 0.0);
  bool orthed = false;
  int idx=0, idx_conj=0, k=0;
  while(!orthed && k < k_max) {
    printf("Orthing iter %d\n", k);
    // Compute R_{k}
    for(int b=0; b<block_size; b++) {
      for(int c=0; c<b; c++) {
	Complex cnorm = cDotProd(vecs[c], vecs[b]);
	caxpy(-cnorm, vecs[c], vecs[b]);
	
	idx      = b*block_size + c;
	idx_conj = c*block_size + b;
	
	tmp[idx     ] = cnorm;
	tmp[idx_conj] = 0.0;
      }
      tmp[b*(block_size + 1)] = normalise(vecs[b]);
    }
    // Accumulate R_{k}
    updateBlockBeta(tmp, beta, k, block_offset, block_size);
    orthed = orthoCheck(vecs, block_size, false);
    k++;
  }
}

//Gram-Schmidt the input vectors
void gramSchmidt(std::vector<Complex*> &vecs) {
  
  int size = (int)vecs.size();
  for(int i=0; i<size; i++) {
    for(int j=0; j<i; j++) {
      Complex cnorm = cDotProd(vecs[j], vecs[i]);
      caxpy(-cnorm, vecs[j], vecs[i]);      
    }
    normalise(vecs[i]);    
  }
}

