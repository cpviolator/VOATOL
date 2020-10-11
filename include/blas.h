#pragma once

#include <voatol_internal.h>

extern int mat_size;
extern bool verbose;

//Simple Complex Linear Algebra Helpers
void zero(Complex *x);
void copy(Complex *x, Complex *y);
void ax(double a, Complex *x);
void cax(Complex a, Complex *x);
void axpy(double a, const Complex *x, Complex *y);
void caxpy(Complex a, const Complex *x, Complex *y);
void axpby(double a, Complex *x, double b, Complex *y);
void caxpby(Complex a, Complex *x, Complex b, Complex *y);
Complex dotProd(Complex *x, Complex *y);
Complex dotProdMatVec(Complex *x, Complex *y);
Complex cDotProd(const Complex *x, const Complex *y);
double norm2(Complex *x);
double norm(Complex *x);
double normalise(Complex *x);

//Orthogonalise r against the j vectors in vectorSpace
void orthogonalise(Complex *r, std::vector<Complex*> vectorSpace, int j);

// Matrix * vector operation 
void matVec(Complex **mat, Complex *out, Complex *in);

// Chebyshev operator
void chebyOp(Complex **mat, Complex *out, Complex *in, double a, double b, int poly_deg);

// Measure the deviation from orthogonality of r[k] with vecs[i],
// place results in s[i*block_size + k]
void measOrthoDev(std::vector<Complex*> vecs, std::vector<Complex*> r, std::vector<Complex> &s, int j);

// Check that orthonormality is preserved
bool orthoCheck(std::vector<Complex*> vecs, int size);

void updateBlockBeta(std::vector<Complex> tmp, std::vector<Complex> &beta, int k, int block_offset, int block_size);

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
			  int block_offset, int k_max);

//Gram-Schmidt the input vectors
void gramSchmidt(std::vector<Complex*> &vecs);

