#pragma once

extern int mat_size;
extern bool verbose;

#include "voatol_internal.h"

void saveTRLMSolverState(Complex **mat, const std::vector<Complex*> kSpace, const std::vector<double> alpha, const std::vector<double> beta, const string name);

void loadTRLMSolverState(Complex **mat, std::vector<Complex*> kSpace, std::vector<double> alpha, std::vector<double> beta, const string name);
