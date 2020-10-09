#pragma once

extern int mat_size;
extern bool verbose;

#include "linAlgHelpers.h"

std::vector<double> ritz_mat;
std::vector<Complex> block_ritz_mat;

//Functions used in the lanczos algorithm
//---------------------------------------
void iterRefineReal(std::vector<Complex*> &kSpace, std::vector<Complex*> &r, std::vector<double> &alpha, std::vector<double> &beta, int j);

// Block caxpy r[k] = r[k] + s_{k,j} * vecs_{j} 
void CAXPY(std::vector<Complex*> vecs, std::vector<Complex*> &r, std::vector<Complex> &s, int j, bool plus);

void iterRefineBlock(std::vector<Complex*> &kSpace, std::vector<Complex*> &r, std::vector<Complex> &alpha, std::vector<Complex> &beta, int j);

//The Engine
void lanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		 std::vector<double> &beta, std::vector<double> &alpha,
		 std::vector<Complex*> &r, int num_keep, int j,
		 double a_min, double a_max, int poly_deg);

void arnoldiStep(Complex **mat, std::vector<Complex*> &kSpace,
		 Eigen::MatrixXcd &upperHessEigen,
		 std::vector<Complex*> &r, double &beta, int j);

void reorder(std::vector<Complex*> &kSpace, std::vector<Complex> evals, std::vector<double> residua, int nKr, int spectrum);

void reorder(std::vector<Complex*> &kSpace, std::vector<double> alpha, int nKr, bool reverse);

void blockLanczosStep(Complex **mat, std::vector<Complex*> &kSpace,
		      std::vector<Complex> &beta, std::vector<Complex> &alpha,
		      std::vector<Complex*> &r, int num_keep, int j, int block_size,
		      double a_min, double a_max, int poly_deg);

void eigensolveFromUpperHess(MatrixXcd &upperHessEigen, MatrixXcd &Qmat,
			    std::vector<Complex> &evals,
			    std::vector<double> &residua,
			     const double beta, int nKr);

void qriteration(MatrixXcd &Rmat, MatrixXcd &Qmat, const int nKr, const double tol);

int qrFromUpperHess(MatrixXcd &upperHess, MatrixXcd &Qmat, std::vector<Complex> &evals,
		    std::vector<double> &residua, const double beta, const int nKr,
		    const double tol);


void eigensolveFromArrowMat(int num_locked, int arrow_pos, int nKr, std::vector<double> &alpha, std::vector<double> &beta, std::vector<double> &residua, bool reverse); 

void eigensolveFromBlockArrowMat(int num_locked, int arrow_pos, int nKr, int block_size, int restart_iter, std::vector<Complex> &alpha, std::vector<Complex> &beta, std::vector<Complex> &arrow_eigs, std::vector<double> &residua, bool reverse);

void computeEvals(Complex **mat, std::vector<Complex*> &kSpace, std::vector<double> &residua, std::vector<Complex> &evals, int nEv);
  

void rotateVecs(std::vector<Complex*> &vecs, Eigen::MatrixXd mat, int num_locked, int iter_keep, int dim);


void rotateVecsComplex(std::vector<Complex*> &vecs, Eigen::MatrixXcd mat, int num_locked, int iter_keep, int dim);

void computeKeptRitz(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, std::vector<double> &beta);
  
void computeKeptRitzComplex(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, int block_size, std::vector<Complex> &beta);
  

void permuteVecs(std::vector<Complex*> &kSpace, Eigen::MatrixXd mat, int num_locked, int size);

void computeKeptRitzLU(std::vector<Complex*> &kSpace, int nKr, int num_locked, int iter_keep, int batch, std::vector<double> &beta, int iter);
  
