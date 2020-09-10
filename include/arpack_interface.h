#pragma once

void arpackErrorHelpSAUPD(int *iparam_);
void arpackErrorHelpSEUPD(int *iparam_);

#define ARPACK(s) s ## _

#ifdef __cplusplus
extern "C" {
#endif

//  Interface functions to the external ARPACK library. These functions utilize 
//  ARPACK's implemntation of the Implicitly Restarted Arnoldi Method to compute a 
//  number of eigenvectors/eigenvalues with user specified features, such as those 
//  with small real part, small magnitude etc. Parallel (OMP/MPI) versions
//  are also supported.
  
// Serial, double prec complex eigenvectors
  extern int ARPACK(znaupd)(int *ido, char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid,
			    int *ncv, std::complex<double> *v, int *ldv, int *iparam, int *ipntr,
			    std::complex<double> *workd, std::complex<double> *workl, int *lworkl, double *rwork,
			    int *info);

// Serial, double prec complex eigenvalues
  extern int ARPACK(zneupd)(int *comp_evecs, char *howmany, int *select, std::complex<double> *evals,
			    std::complex<double> *v, int *ldv, std::complex<double> *sigma, std::complex<double> *workev,
			    char *bmat, int *n, char *which, int *nev, double *tol, std::complex<double> *resid, int *ncv,
			    std::complex<double> *v1, int *ldv1, int *iparam, int *ipntr, std::complex<double> *workd,
			    std::complex<double> *workl, int *lworkl, double *rwork, int *info, int howmany_size,
			    int bmat_size, int spectrum_size);
  
  
  extern int ARPACK(mcinitdebug)(int*,int*,int*,int*,int*,int*,int*,int*);
    
  //ARPACK initlog and finilog routines for printing the ARPACK log  
  extern int ARPACK(initlog) (int*, char*, int);
  extern int ARPACK(finilog) (int*);
  
#ifdef __cplusplus
}
#endif


void arpackErrorHelpSAUPD(int *iparam_) {
  printf("\nError help NAUPD\n\n");
  printf("INFO Integer.  (INPUT/OUTPUT)\n");
  printf("     If INFO .EQ. 0, a randomly initial residual vector is used.\n");
  printf("     If INFO .NE. 0, RESID contains the initial residual vector,\n");
  printf("                        possibly from a previous run.\n");
  printf("     Error flag on output.\n");
  printf("     =  0: Normal exit.\n");
  printf("     =  1: Maximum number of iterations taken.\n");
  printf("        All possible eigenvalues of OP has been found. IPARAM(5)\n");
  printf("        returns the number of wanted converged Ritz values.\n");
  printf("     =  2: No longer an informational error. Deprecated starting\n");
  printf("        with release 2 of ARPACK.\n");
  printf("     =  3: No shifts could be applied during a cycle of the\n");
  printf("        Implicitly restarted Arnoldi iteration. One possibility\n");
  printf("        is to increase the size of NCV relative to NEV.\n");
  printf("        See remark 4 below.\n");
  printf("     = -1: N must be positive.\n");
  printf("     = -2: NEV must be positive.\n");
  printf("     = -3: NCV-NEV >= 1 and less than or equal to N.\n");
  printf("     = -4: The maximum number of Arnoldi update iteration\n");
  printf("        must be greater than zero.\n");
  printf("     = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
  printf("     = -6: BMAT must be one of 'I' or 'G'.\n");
  printf("     = -7: Length of private work array is not sufficient.\n");
  printf("     = -8: Error return from LAPACK eigenvalue calculation;\n");
  printf("     = -9: Starting vector is zero.\n");
  printf("     = -10: IPARAM(7) must be 1,2,3.\n");
  printf("     = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
  printf("     = -12: IPARAM(1) must be equal to 0 or 1.\n");
  printf("     = -9999: Could not build an Arnoldi factorization.\n");
  printf("        User input error highly likely.  Please\n");
  printf("        check actual array dimensions and layout.\n");
  printf("        IPARAM(5) returns the size of the current Arnoldi\n");
  printf("        factorization.\n");
  printf("        iparam_[5] = %d\n", iparam_[4]);
}

void arpackErrorHelpSEUPD(int *iparam_) {
  printf("\nError help NEUPD\n\n");
  printf("INFO Integer.  (OUTPUT)\n");
  printf("     Error flag on output.\n");
  printf("     =  0: Normal exit.\n");
  printf("     =  1: The Schur form computed by LAPACK routine csheqr\n");
  printf("        could not be reordered by LAPACK routine ztrsen.\n");
  printf("        Re-enter subroutine zneupd with IPARAM(5)=NCV and\n");
  printf("        increase the size of the array D to have\n");
  printf("        dimension at least dimension NCV and allocate at\n");
  printf("        least NCV\n");
  printf("        columns for Z. NOTE: Not necessary if Z and V share\n");
  printf("        the same space. Please notify the authors if this\n");
  printf("        error occurs.\n");
  printf("     = -1: N must be positive.\n");
  printf("     = -2: NEV must be positive.\n");
  printf("     = -3: NCV-NEV >= 1 and less than or equal to N.\n");
  printf("     = -5: WHICH must be 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'\n");
  printf("     = -6: BMAT must be one of 'I' or 'G'.\n");
  printf("     = -7: Length of private work WORKL array is inufficient.\n");
  printf("     = -8: Error return from LAPACK eigenvalue calculation.\n");
  printf("        This should never happened.\n");
  printf("     = -9: Error return from calculation of eigenvectors.\n");
  printf("        Informational error from LAPACK routine ztrevc.\n");
  printf("     = -10: IPARAM(7) must be 1,2,3\n");
  printf("     = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.\n");
  printf("     = -12: HOWMNY = 'S' not yet implemented\n");
  printf("     = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true.\n");
  printf("     = -14: ZNAUPD did not find any eigenvalues to sufficient\n");
  printf("        accuracy.\n");
  printf("     = -15: ZNEUPD got a different count of the number of\n");
  printf("        converged Ritz values than ZNAUPD got. This\n");
  printf("        indicates the user probably made an error in\n");
  printf("        passing data from ZNAUPD to ZNEUPD or that the\n");
  printf("        data was modified before entering ZNEUPD\n");
  printf("        iparam_[5] = %d\n", iparam_[4]);
}

