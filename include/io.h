#pragma once

extern int mat_size;
extern bool verbose;

//#include "linAlgHelpers.h"

void saveTRLMSolverState(Complex **mat, const std::vector<Complex*> kSpace, const std::vector<double> alpha, const std::vector<double> beta, const string name) {

  // Open file
  fstream outPutFile;
  outPutFile.open(name,ios::in|ios::out|ios::trunc);  
  outPutFile.setf(ios_base::fixed,ios_base::floatfield); 

  // Save operator
  for(int i=0; i<mat_size; i++) {
    for(int j=0; j<mat_size; j++) {
      outPutFile << setprecision(16) << setw(20) << mat[i][j].real() << endl;
      outPutFile << setprecision(16) << setw(20) << mat[i][j].imag() << endl;
    }
  }

  // Save Krylov space
  for(int n=0; n<(int)kSpace.size(); n++) {
    for(int i=0; i<mat_size; i++) {
      outPutFile << setprecision(16) <<  setw(20) << kSpace[n][i].real() << endl;
      outPutFile << setprecision(16) <<  setw(20) << kSpace[n][i].imag() << endl;
    }
  }

  // Save tridiag components
  for(int i=0; i<(int)alpha.size(); i++) outPutFile << setprecision(16) <<  setw(20) << alpha[i] << endl;
  for(int i=0; i<(int)beta.size()-1; i++) outPutFile << setprecision(16) <<  setw(20) << beta[i] << endl;    
  outPutFile.close();
}

void loadTRLMSolverState(Complex **mat, std::vector<Complex*> kSpace, std::vector<double> alpha, std::vector<double> beta, const string name) {

  fstream inPutFile;
  inPutFile.open(name);
  string val;
  if(!inPutFile.is_open()) {
    cout << "Error opening file " << name << endl;
    exit(0);
  }

  // Read operator
  for(int i=0; i<mat_size; i++) {
    for(int j=0; j<mat_size; j++) {	    
      getline(inPutFile, val);
      mat[i][j].real(stod(val));
      getline(inPutFile, val);
      mat[i][j].imag(stod(val));
    }
  }

  // Read Krylov space
  for(int n=0; n<(int)kSpace.size(); n++) {
    for(int i=0; i<mat_size; i++) {
      getline(inPutFile, val);
      kSpace[n][i].real(stod(val));
      getline(inPutFile, val);
      kSpace[n][i].imag(stod(val));
    }
  }

  // Read tridiag componenets
  for(int i=0; i<(int)alpha.size(); i++) {
    getline(inPutFile, val);
    alpha[i] = stod(val);
  }
  for(int i=0; i<(int)beta.size()-1; i++) {
    getline(inPutFile, val);
    beta[i] = stod(val);
  }
  inPutFile.close();
}
