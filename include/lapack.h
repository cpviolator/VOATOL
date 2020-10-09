#pragma once

extern int mat_size;

void zscal(int n, Complex da, Complex *X, int incx) {
  //Scales a vector by a constant
  for(int i=0; i<n; i+=incx) {
    X[i] *= da;
  }  
}

void zdscal(int n, double da, Complex *X, int incx) {
  //Scales a vector by a constant
  for(int i=0; i<n; i+=incx) {
    X[i] *= da;
  }  
}

void zlascl(double cfrom, double cto, Complex *X) {

  //ZLASCL multiplies the M by N complex matrix A by the real scalar  
  //CTO/CFROM.  This is done without over/underflow as long as the final
  //result CTO*A(I,J)/CFROM does not over/underflow.
  
  //Get machine parameters

  double smlnum = DBL_MIN;
  double bignum = 1.0 / smlnum;

  double cfromc = cfrom, ctoc = cto, cfrom1 = 0, cto1 = 0, mul = 0;
  
  bool done = false;
  
  while(!done) {
    cfrom1 = cfromc*smlnum;
    cto1 = ctoc / bignum;
    if(abs( cfrom1 ) > abs( ctoc ) && ctoc != 0.0 ) {
      mul = smlnum;
      done = false;
      cfromc = cfrom1;
    } else if(abs( cto1 ) > abs( cfromc )) {
      mul = bignum;
      done = false;
      ctoc = cto1;
    } else {
      mul = ctoc / cfromc;
      done = true;
    }
    
    for(int j=0; j < mat_size; j++) X[j] *= mul;
  }
}

double dlapy2(double X, double Y) {
      
  //DLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary overflow.
  double XABS = abs( X );
  double YABS = abs( Y );
  double dlapy2;
  double W = std::max(XABS, YABS);
  double Z = std::min(XABS, YABS);
  if( Z == 0.0 ) {
    dlapy2 = W;
  } else {
    dlapy2 = W*sqrt( 1.0 + (Z/W)*(Z/W) );
  }
  return dlapy2;
}

double dlapy3(Complex X, Complex Y, Complex Z) {

  //DLAPY3 returns sqrt(x**2+y**2+z**2), taking care not to cause unnecessary overflow.
  double XABS = abs( X );
  double YABS = abs( Y );
  double ZABS = abs( Z );
  double dlapy3;
  double W = std::max(std::max(XABS, YABS), ZABS);
  if( W == 0.0 ) {
    dlapy3 = 0.0;
  } else {
    dlapy3 = W*sqrt( (XABS/W)*(XABS/W) + (YABS/W)*(YABS/W) + (ZABS/W)*(ZABS/W) );
  }
  return dlapy3;
}
  
double dznrm2(int N, Complex *X, int INCX) {
  
  //DZNRM2 returns the euclidean norm of a vector via the function
  //name, so that

  //DZNRM2 := sqrt( conjg( x' )*x )

  double NORM, SCALE, SSQ, TEMP;
    
  if( N < 1 || INCX < 1 ) {
    NORM = 0.0;
  } else {
    SCALE = 0.0;
    SSQ   = 1.0;
    
    //The following loop is equivalent to this call to the LAPACK
    //auxiliary routine:
    //CALL ZLASSQ( N, X, INCX, SCALE, SSQ )
    
    for(int i=0; i < 1 + ( N - 1 )*INCX; i += INCX) {
      if( X[i].real() != 0.0 ) {
	TEMP = abs(X[i].real());
	if(SCALE < TEMP) {	  
	  SSQ = 1.0 + SSQ*((SCALE*SCALE)/(TEMP*TEMP));
	  SCALE = TEMP;
	} else {
	  SSQ = SSQ + ((TEMP*TEMP)/(SCALE*SCALE));
	}
      }
      if(X[i].imag() != 0.0) {	
	TEMP = abs(X[i].imag());
	if(SCALE < TEMP){	  
	  SSQ = 1.0 + SSQ*((SCALE*SCALE)/(TEMP*TEMP));
	  SCALE = TEMP;
	} else {
	  SSQ = SSQ + ((TEMP*TEMP)/(SCALE*SCALE));
	}
      }
    }
    NORM = SCALE * sqrt( SSQ );
  }
  return NORM;
}

double zlanhs(Eigen::MatrixXcd A) {
  
  //Find norm1(A)
  double value = 0.0;
  
  for(int j = 0; j<A.rows(); j++) {
    double sum = 0.0;
    for (int i = 0; i < A.cols(); i++) {
      sum += abs(A(i,j));
    }
    value = std::max(value, sum);
  }
  return value;
}

double cabs1(Complex C) {
  return abs(C.real()) + abs(C.imag());
}

// OK Nov 17 12:51
void zlartg(const Complex F, const Complex G, std::vector<double> &cos,
	    std::vector<Complex> &sin, std::vector<Complex> &r) {  
  
  // Compute Givens Rotations. Adapted from zlartg.f
  // ZLARTG generates a plane rotation so that
  // 
  //      [  CS  SN  ]     [ F ]     [ R ]
  //      [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
  //      [ -SN  CS  ]     [ G ]     [ 0 ]
  
  Complex cZero(0.0,0.0);
  Complex SS, GS, FS;
  double G2, F2, D, DI, FA, GA;
  
  if(G == cZero) {
    r[0] = F;
    cos[0] = 1.0;
    sin[0] = cZero;
  } else if (F == cZero) {
    cos[0] = 0.0;
    sin[0] = conj(G)/abs(G);
    r[0] = abs(G);
  } else {
    double F1 = abs(F.real()) + abs(F.imag());
    double G1 = abs(G.real()) + abs(G.imag());    
    if (F1 >= G1) {
      GS = G / F1;
      G2 = pow(hypot(GS.real(), GS.imag()), 2);
      FS = F / F1;
      F2 = pow(hypot(FS.real(), FS.imag()), 2);
      D = hypot(1.0, abs(GS)/abs(FS));
      cos[0] = 1.0/D;
      sin[0] = conj(GS) * FS * (cos[0] / F2);
      r[0] = F*D;
    } else {
      FS = F / G1;
      F2 = pow(hypot(FS.real(), FS.imag()),2);
      FA = sqrt(F2);
      GS = G / G1;
      G2 = pow(hypot(GS.real(), GS.imag()),2);
      GA = sqrt(G2);
      D = hypot(1.0, abs(FS)/abs(GS));
      DI = 1.0/D;
      cos[0] = (FA / GA) * DI;
      SS = (conj(GS) * FS)/(FA*GA);
      sin[0] = SS * DI;
      r[0] = G * SS * D;
    }
  }
}


// OK Nov 17 12:51
void zlartg(const Complex F, const Complex G, double &cos, Complex &sin, Complex &r)
{  
  
  // Compute Givens Rotations. Adapted from zlartg.f
  // ZLARTG generates a plane rotation so that
  // 
  //      [  CS  SN  ]     [ F ]     [ R ]
  //      [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1.
  //      [ -SN  CS  ]     [ G ]     [ 0 ]
  
  Complex cZero(0.0,0.0);
  Complex SS, GS, FS;
  double G2, F2, D, DI, FA, GA;
  
  if(G == cZero) {
    r = F;
    cos = 1.0;
    sin = cZero;
  } else if (F == cZero) {
    cos = 0.0;
    sin = conj(G)/abs(G);
    r = abs(G);
  } else {
    double F1 = abs(F.real()) + abs(F.imag());
    double G1 = abs(G.real()) + abs(G.imag());    
    if (F1 >= G1) {
      GS = G / F1;
      G2 = pow(hypot(GS.real(), GS.imag()), 2);
      FS = F / F1;
      F2 = pow(hypot(FS.real(), FS.imag()), 2);
      D = hypot(1.0, abs(GS)/abs(FS));
      cos = 1.0/D;
      sin = conj(GS) * FS * (cos / F2);
      r = F*D;
    } else {
      FS = F / G1;
      F2 = pow(hypot(FS.real(), FS.imag()),2);
      FA = sqrt(F2);
      GS = G / G1;
      G2 = pow(hypot(GS.real(), GS.imag()),2);
      GA = sqrt(G2);
      D = hypot(1.0, abs(FS)/abs(GS));
      DI = 1.0/D;
      cos = (FA / GA) * DI;
      SS = (conj(GS) * FS)/(FA*GA);
      sin = SS * DI;
      r = G * SS * D;
    }
  }
}


void zsortc(int which, int n, std::vector<Complex> &x, std::vector<Complex> &y) {
  
  /*
    c\BeginDoc
    c
    c\Name: zsortc
    c
    c\Description:
    c  Sorts the Complex*16 arrays in X and Y into the order 
    c  specified by WHICH.
    c
    c\Arguments
    c  which   Character*2.  (Input)
    c        0  'LM' -> sort into increasing order of magnitude.
    c        1  'SM' -> sort into decreasing order of magnitude.
    c        2  'LR' -> sort with real(x) in increasing algebraic order 
    c        3  'SR' -> sort with real(x) in decreasing algebraic order
    c        4  'LI' -> sort with imag(x) in increasing algebraic order
    c        5  'SI' -> sort with imag(x) in decreasing algebraic order
    c
    c  n       Integer.  (INPUT)
    c          Size of the arrays.
    c
    c  x       Complex*16 array of length N.  (INPUT/OUTPUT)
    c          This is the array to be sorted.
    c
    c  y       Complex*16 array of length N.  (INPUT/OUTPUT)
    c
    c\EndDoc
    c
    c-----------------------------------------------------------------------
  */

  std::vector<std::pair<Complex, Complex>> array(n);
  for(int i=0; i<n; i++) array[i] = std::make_pair(x[i], y[i]);
  
  switch(which) {
  case 0: std::sort(array.begin(), array.begin()+n,
		    [] (const pair<Complex,Complex> &a,
			const pair<Complex,Complex> &b) {
		      return (abs(a.first) < abs(b.first)); } );
    break;
  case 1: std::sort(array.begin(), array.begin()+n,
		    [] (const pair<Complex,Complex> &a,
			const pair<Complex,Complex> &b) {
		      return (abs(a.first) > abs(b.first)); } );
    break;
  case 2: std::sort(array.begin(), array.begin()+n,
		    [] (const pair<Complex,Complex> &a,
			const pair<Complex,Complex> &b) {
		      return (a.first).real() < (b.first).real(); } );
    break;
  case 3: std::sort(array.begin(), array.begin()+n,
		    [] (const pair<Complex,Complex> &a,
			const pair<Complex,Complex> &b) {
		      return (a.first).real() > (b.first).real(); } );
    break;
  case 4: std::sort(array.begin(), array.begin()+n,
		    [] (const pair<Complex,Complex> &a,
			const pair<Complex,Complex> &b) {
		      return (a.first).imag() < (b.first).imag(); } );
    break;
  case 5: std::sort(array.begin(), array.begin()+n,
		    [] (const pair<Complex,Complex> &a,
			const pair<Complex,Complex> &b) {
		      return (a.first).imag() > (b.first).imag(); } );
    break;
  default: cout << "Undefined sort" << endl;
  }

  // Repopulate x and y arrays with sorted elements
  for(int i=0; i<n; i++) {
    x[i] = array[i].first;
    y[i] = array[i].second;
  }
}

// Overloaded version of zsortc to deal with real y array.
void zsortc(int which, int n, std::vector<Complex> &x, std::vector<double> &y) {

  std::vector<Complex> y_tmp(n,0.0);
  for(int i=0; i<n; i++) y_tmp[i].real(y[i]);
  zsortc(which, n, x, y_tmp);
  for(int i=0; i<n; i++) y[i] = y_tmp[i].real();
}

// Overloaded version of zsortc to deal with real x array.
void zsortc(int which, int n, std::vector<double> &x, std::vector<Complex> &y) {

  std::vector<Complex> x_tmp(n,0.0);
  for(int i=0; i<n; i++) x_tmp[i].real(x[i]);
  zsortc(which, n, x_tmp, y);
  for(int i=0; i<n; i++) x[i] = x_tmp[i].real();
}

// Overloaded version of zsortc to deal with real x and y array.
void zsortc(int which, int n, std::vector<double> &x, std::vector<double> &y) {

  std::vector<Complex> x_tmp(n,0.0);
  std::vector<Complex> y_tmp(n,0.0);
  for(int i=0; i<n; i++) {
    x_tmp[i].real(x[i]);
    y_tmp[i].real(y[i]);
  }
  zsortc(which, n, x_tmp, y_tmp);
  for(int i=0; i<n; i++) {
    x[i] = x_tmp[i].real();
    y[i] = y_tmp[i].real();
  }
}



void zlarf(int SIDE, int M, int N, Complex TAU, Complex *C, int LDC) {

  /*
   *  ZLARF applies a complex elementary reflector H to a complex M-by-N
   *  matrix C, from either the left or the right. H is represented in the
   *  form
   *
   *        H = I - tau * v * v'
   *
   *  where tau is a complex scalar and v is a complex vector.
   *
   *  If tau = 0, then H is taken to be the unit matrix.
   *
   *  To apply H' (the conjugate transpose of H), supply conjg(tau) instead
   *  tau.
   *
   *  Arguments
   *  =========
   *
   *  SIDE    (input) CHARACTER*1
   *          = 'L': form  H * C
   *          = 'R': form  C * H
   *
   *  M       (input) INTEGER
   *          The number of rows of the matrix C.
   *
   *  N       (input) INTEGER
   *          The number of columns of the matrix C.
   *
   *  V       (input) COMPLEX*16 array, dimension
   *                     (1 + (M-1)*abs(INCV)) if SIDE = 'L'
   *                  or (1 + (N-1)*abs(INCV)) if SIDE = 'R'
   *          The vector v in the representation of H. V is not used if
   *          TAU = 0.
   *
   *  INCV    (input) INTEGER
   *          The increment between elements of v. INCV <> 0.
   *
   *  TAU     (input) COMPLEX*16
   *          The value tau in the representation of H.
   *
   *  C       (input/output) COMPLEX*16 array, dimension (LDC,N)
   *          On entry, the M-by-N matrix C.
   *          On exit, C is overwritten by the matrix H * C if SIDE = 'L',
   *          or C * H if SIDE = 'R'.
   *
   *  LDC     (input) INTEGER
   *          The leading dimension of the array C. LDC >= max(1,M).
   *
   */

  Complex cZero(0.0,0.0);
  
  if(SIDE == 0) {
    // Form  H * C
    if(TAU != cZero) {
      //w := C' * v
    }
  }
}

void zlarfg(int N, Complex &ALPHA, Complex *X, int INCX, Complex &TAU) {
  
  /*  
    ZLARFG generates a complex elementary reflector H of order n, such
    that
    
    H' * ( alpha ) = ( beta ),   H' * H = I.
         (   x   )   (   0  )
	 
    where alpha and beta are scalars, with beta real, and x is an
    (n-1)-element complex vector. H is represented in the form
	 
    H = I - tau * ( 1 ) * ( 1 v' ) ,
                  ( v )
		  
    where tau is a complex scalar and v is a complex (n-1)-element
    vector. Note that H is not hermitian.

    If the elements of x are all zero and alpha is real, then tau = 0
    and H is taken to be the unit matrix.
    
    Otherwise  1 <= real(tau) <= 2  and  abs(tau-1) <= 1 .
    
    Arguments
    =========
    
    N       (input) INTEGER
    The order of the elementary reflector.
    
    ALPHA   (input/output) COMPLEX*16
    On entry, the value alpha.
    On exit, it is overwritten with the value beta.*

    X       (input/output) COMPLEX*16 array, dimension
    (1+(N-2)*abs(INCX))
    On entry, the vector x.
    On exit, it is overwritten with the vector v.
    
    INCX    (input) INTEGER
    The increment between elements of X. INCX > 0.
    
    TAU     (output) COMPLEX*16
    The value tau.
    
    =====================================================================
  */
  
  int KNT;
  double ALPHI, ALPHR, BETA, RSAFMN, SAFMIN, XNORM;

  Complex cOne(1.0,0.0);
  
  if( N <= 0 ) {
    TAU = 0.0;
    return;
  }

  XNORM = dznrm2(N-1, X, INCX);
  ALPHR = ALPHA.real();
  ALPHI = ALPHA.imag();
  
  if(XNORM == 0.0 && ALPHI == 0.0 ) {
    //H  =  I
    TAU = 0.0;
  } else {
    //general case
    int sign1 = (ALPHR > 0) - (ALPHR < 0);
    BETA = -sign1 * dlapy3(ALPHR, ALPHI, XNORM);
    SAFMIN = DBL_MIN / DBL_EPSILON;
    RSAFMN = 1.0 / SAFMIN;

    if(abs( BETA ) < SAFMIN ) {
      //XNORM, BETA may be inaccurate; scale X and recompute them      
      KNT = 0;
      while (abs( BETA ) < SAFMIN) {
	KNT++;
	zdscal(N-1, RSAFMN, X, INCX);
	BETA = BETA*RSAFMN;
	ALPHI = ALPHI*RSAFMN;
	ALPHR = ALPHR*RSAFMN;
      }
      
      //New BETA is at most 1, at least SAFMIN
      
      XNORM = dznrm2( N-1, X, INCX );
      ALPHA.real(ALPHR); ALPHA.imag(ALPHI);
      int sign2 = (ALPHR > 0) - (ALPHR < 0);
      BETA = -sign2 * dlapy3( ALPHR, ALPHI, XNORM );
      TAU.real( (BETA-ALPHR ) / BETA); TAU.imag(-ALPHI / BETA);
      ALPHA = cOne / (ALPHA-BETA);
      zscal( N-1, ALPHA, X, INCX );

      //If ALPHA is subnormal, it may lose relative accuracy
  
      ALPHA = BETA;
      for(int j = 0; j < KNT; j++) {
	ALPHA = ALPHA*SAFMIN;
      }
    } else {
      TAU.real(( BETA-ALPHR ) / BETA); TAU.imag( -ALPHI / BETA );
      ALPHA = cOne / (ALPHA-BETA);
      zscal( N-1, ALPHA, X, INCX );
      ALPHA = BETA;
    }
  }
}

void zgeqr2(int M, int N, Eigen::MatrixXcd &A, Complex *tau) {

  /*
   *  ZGEQR2 computes a QR factorization of a complex m by n matrix A:
   *  A = Q * R.
   *
   *  Arguments
   *  =========
   *
   *  M       (input) INTEGER
   *          The number of rows of the matrix A.  M >= 0.
   *
   *  N       (input) INTEGER
   *          The number of columns of the matrix A.  N >= 0.
   *
   *  A       (input/output) COMPLEX*16 array, dimension (LDA,N)
   *          On entry, the m by n matrix A.
   *          On exit, the elements on and above the diagonal of the array
   *          contain the min(m,n) by n upper trapezoidal matrix R (R is
   *          upper triangular if m >= n); the elements below the diagonal,
   *          with the array TAU, represent the unitary matrix Q as a
   *          product of elementary reflectors (see Further Details).
   *
   *  LDA     (input) INTEGER
   *          The leading dimension of the array A.  LDA >= max(1,M).
   *
   *  TAU     (output) COMPLEX*16 array, dimension (min(M,N))
   *          The scalar factors of the elementary reflectors (see Further
   *          Details).
   *
   *  WORK    (workspace) COMPLEX*16 array, dimension (N)
   *
   *  INFO    (output) INTEGER
   *          = 0: successful exit
   *          < 0: if INFO = -i, the i-th argument had an illegal value
   *
   *  Further Details
   *  ===============
   *
   *  The matrix Q is represented as a product of elementary reflectors
   *
   *     Q = H(1) H(2) . . . H(k), where k = min(m,n).
   *
   *  Each H(i) has the form
   *
   *     H(i) = I - tau * v * v'
   *
   *  where tau is a complex scalar, and v is a complex vector with
   *  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
   *  and tau in TAU(i).
   *
   *  =====================================================================
   */

  Complex cOne(1.0,0.0);
  int K = std::min(M,N);

  for( int i=0; i<K; i++) {

    cout <<"I="<<i<<endl;
          
    //Generate elementary reflector H(i) to annihilate A(i+1:m,i)
    Complex X[M-(i+1)];
    int start = std::min(i+1,M);
    for(int j=0; j<M-(i+1); j++) X[j] = A(start+j,i);
    cout << "ZGEQR2: calling ZLARFG" << endl;
    cout << "M-I = " << M-i << endl;

    cout << "Aii pre  = " << A(i,i) << endl;
    Complex Aiipre = A(i,i);
    cout << "tau pre  = " << tau[i] << endl;
    zlarfg(M-i, A(i,i), X, 1, tau[i]);
    cout << "Aii post = " << A(i,i) << endl;
    cout << "tau post = " << tau[i] << endl;
    cout << "tau - A(i,i) pre = " << tau[i] - Aiipre << endl;
    
    for(int l=0; l<M; l++)
      for(int j=0; j<M; j++)
	cout << "("<<j<<","<<l<<") = "<<A(j,l)<<endl;
    
    
    if(i<N-1) {
      
      //Apply H(i)' to A(i:m,i+1:n) from the left
      //Emulate zlarf.f
      
      Complex ALPHA = A(i,i);
      A(i,i) = cOne;
      cout << "ZGEQR2: calling ZLARF" << endl;
      cout << "M-I = " << M-i << endl;
      cout << "N-(I+1) = " << N-(i+1) << endl;
      cout << "LDA = " << M << endl;

      for(int j=i; j<M; j++)
	cout << "VEC("<<j<<","<<i<<") = "<<A(j,i)<<endl;
      
      // Get the A(i,i) column
      Eigen::VectorXcd V = A.col(i).tail(M-i);
      //cout << "1" << endl;
      // Get the A sub block.
      Eigen::MatrixXcd Asub = A.block(i,i+1,M-i,N-(i+1));
      //cout << "2" << endl;
      Eigen::MatrixXcd AsubA = Asub.adjoint();
      
      cout << A.rows() << " " << A.cols() << endl;
      cout << Asub.rows() << " " << Asub.cols() << endl;
      cout << AsubA.rows() << " " << AsubA.cols() << endl;
      
      //w := C' * v
      Asub.adjointInPlace();
      Eigen::VectorXcd W = Asub * V;
      //cout << "3" << endl;      
      
      cout << W.rows() << " " << V.rows() << endl;
      
      //C := C - v * w'
      for(int n=0; n<(N-(i+1)); n++) {
	cout << -conj(tau[i]) << " " << conj(W(n)) << endl;;
	for(int m=0; m<(M-i); m++) {
	  cout << "gerc A("<<m+i<<","<<(i+1)+n<<") = "<<A(i+m,(i+1)+n)<<endl;
	  cout << V(m) << " " << -conj(tau[i]) * conj(W(n)) << endl;
	  if(tau[i] != 0.0) A(i+m,(i+1)+n) += -conj(tau[i]) * V(m) * conj(W(n)); 
	}
      }      
      A(i,i) = ALPHA;      
    }
  }
}

void znapps(int num_keep, int nshifts, std::vector<Complex> ritz_vals, Eigen::MatrixXcd &upperHess, Eigen::MatrixXcd &Qmat) {

  bool znapps_verbose = true;
  
  int dim = num_keep + nshifts;
  
  for(int jj=0; jj<nshifts; jj++) {

    Complex sigma = ritz_vals[jj];
    int istart = 0;
    int iend = 0;
    bool complete = false;
    while(!complete) {
      bool split = false;
      for(int i=istart; i<dim-1 && !split; i++) {

	//          %----------------------------------------%
	//          | Check for splitting and deflation. Use |
	//          | a standard test as in the QR algorithm |
	//          | REFERENCE: LAPACK subroutine zlahqr    |
	//          %----------------------------------------%
	
	if(znapps_verbose) cout << "cabs1" << endl;
	double tst1 = cabs1( upperHess(i,i) ) + cabs1( upperHess(i+1, i+1) );
	if(znapps_verbose) cout  << "cabs1 good" << endl;

	if(znapps_verbose) cout  << "zlanhs" << endl;
	if(tst1 == 0.0) tst1 = zlanhs(upperHess);
	if(znapps_verbose) cout  << "zlanhs good" << endl;
	
	if(abs(upperHess(i+1,i).real()) <= 1e-16) {
	  if(znapps_verbose) {
	    cout  << "Split at i=" << i << endl; exit(0);
	  }
	  iend = i;
	  upperHess(i+1,i) = 0.0;
	  split = true;
	}	
      }
      
      // If there is no split, do the full matrix
      if(!split) iend = dim;

      double c = 0.0;
      Complex s = 0.0, r = 0.0;
      
      Complex h11 = upperHess(istart, istart);
      Complex h21 = upperHess(istart+1, istart);
      Complex f = h11 - sigma;
      Complex g = h21;

      if(znapps_verbose) cout  << "istart = " << istart << " iend-1 = " << iend-1 << endl;      
      for(int i=istart; i<iend-1; i++) { 
	if(znapps_verbose) cout  << "i=" << i << " i+1=" << i+1 << endl;
	//          %------------------------------------------------------%
	//          | Construct the plane rotation G to zero out the bulge |
	//          %------------------------------------------------------%
	
	zlartg(f, g, c, s, r);
	cout << f << " " << g << " " << c << " " << s << " " << r << " " << endl;
	
	if (i > istart) {
	  upperHess(i,i-1) = r;
	  upperHess(i+1,i-1) = 0.0;
	}

	if(znapps_verbose) cout  << "one" << endl;
	//          %---------------------------------------------%
	//          | Apply rotation to the left of H;  H <- G'*H |
	//          %---------------------------------------------%
	
	Complex t = 0.0;
	for(int j=i; j<dim; j++) {
	  t = c*upperHess(i,j) + s*upperHess(i+1,j);
	  upperHess(i+1,j) = -conj(s)*upperHess(i,j) + c*upperHess(i+1,j);
	  upperHess(i,j) = t;
	}
	if(znapps_verbose) cout  << "two" << endl;
	//          %---------------------------------------------%
	//          | Apply rotation to the right of H;  H <- H*G |
	//          %---------------------------------------------%
	
	for(int j=0; j<std::min(i+2, iend); j++) {
	  Complex t = c*upperHess(j,i) + conj(s)*upperHess(j,i+1);
	  upperHess(j,i+1) = -s*upperHess(j,i) + c*upperHess(j,i+1);
	  upperHess(j,i)   = t;
	}
	if(znapps_verbose) cout  << "three" << endl;
	//          %-----------------------------------------------------%
	//          | Accumulate the rotation in the matrix Q;  Q <- Q*G' |
	//          %-----------------------------------------------------%
	
	for(int j=0; j<std::min(i+jj, dim); j++) {
	  Complex t = c*Qmat(j,i) + conj(s)*Qmat(j,i+1);
	  Qmat(j,i+1) = - s*Qmat(j,i) + c*Qmat(j,i+1);
	  Qmat(j,i)   = t;   
	}
	if(znapps_verbose) cout  << "four" << endl;
	//          %---------------------------%
	//          | Prepare for next rotation |
	//          %---------------------------%
	
	if (i < iend-2) {
	  f = upperHess(i+1,i);
	  g = upperHess(i+2,i);
	}
	if(znapps_verbose) cout  << "five" << endl;
	double h_test = (upperHess - upperHess.adjoint()).norm();
	if(znapps_verbose) cout << "H test = " << h_test << endl;

	double q_test = (Qmat - Qmat.adjoint()).norm();
	if(znapps_verbose) cout << "Q test = " << q_test << endl;

	cout << upperHess << endl;
	cout << Qmat << endl;
	
      }
      
      //        %---------------------------------------------------------%
      //        | Apply the same shift to the next block if there is any. |
      //        %---------------------------------------------------------%
      
      istart = iend + 1;
      if(iend == dim) complete = true;
    }
  }  
}

void applyShift(Eigen::MatrixXcd &UH, Eigen::MatrixXcd &Q,		
		int istart, int iend, int nKr, Complex shift, int shift_num, int iter) {
  
  //%------------------------------------------------------%
  //| Construct the plane rotation G to zero out the bulge |
  //%------------------------------------------------------%

  bool ah_debug = false;
  
  std::vector<double> cos(1);
  std::vector<Complex> sin(1);
  std::vector<Complex> r(1);
  
  Complex t;
  Complex cZero(0.0,0.0);  
  Complex f = UH(istart,   istart) - shift;
  Complex g = UH(istart+1, istart);

  for(int i = istart; i < iend-1; i++) {


    if(ah_debug) {
      cout << "f= " << f << endl;
      cout << "g= " << g << endl;      
    }
    
    zlartg(f, g, cos, sin, r);
    if(ah_debug) {
      // Sanity check
      //cout << " shift " << shift << " Sanity["<<i<<"] " << cos[0]*cos[0] << " + " <<  abs(sin[0])*abs(sin[0]) << " = " << cos[0]*cos[0] + pow(sin[0].real(), 2) + pow(sin[0].imag(),2) << endl;
      cout << "r= " << r[0] << endl;
      cout << "c= " << cos[0] << endl;
      cout << "s= " << sin[0] << endl;
      cout << " shift " << shift_num << " sigma " << shift << endl;
      cout << " istart = " << istart << " iend = " << iend << endl;
    }
    if(i > istart) {
      UH(i,i-1) = r[0];
      UH(i+1,i-1) = cZero;	      
    }

    //%---------------------------------------------%
    //| Apply rotation to the left of H;  H <- G'*H |
    //%---------------------------------------------%
    //do 50
    for(int j = i; j < nKr; j++) {
      if(ah_debug) {
	cout<<"pre  h("<<i+1<<","<<j<<")="<<UH(i+1,j)<<endl;
	cout<<"pre  h("<<i<<","<<j<<")="<< UH(i,j)<<endl;
	cout<<"cos=" << cos[0] << " sin = " << sin[0] <<endl;
      }
      t         =  cos[0]       * UH(i,j) + sin[0] * UH(i+1,j);
      UH(i+1,j) = -conj(sin[0]) * UH(i,j) + cos[0] * UH(i+1,j);
      UH(i,j) = t;
      if(ah_debug) {
	cout<<"post h("<<i+1<<","<<j<<")="<<UH(i+1,j)<<endl;
	cout<<"post h("<<i<<","<<j<<")="<< UH(i,j)<<endl;
      }
    }

    //%---------------------------------------------%
    //| Apply rotation to the right of H;  H <- H*G |
    //%---------------------------------------------%
    //do 60
    if(ah_debug) cout << "min60 = min(" << i+1+2 << "," << iend << ") = " << std::min(i+1+2, iend) << endl;
    for(int j = 0; j<std::min(i+1+2, iend); j++) {
      if(ah_debug) {
	cout<<"pre  h("<<j<<","<<i+1<<")="<< UH(j,i+1)<<endl;
	cout<<"pre  h("<<j<<","<<i<<")="<< UH(j,i)<<endl;
	cout<<"cos=" << cos[0] << " sin = " << sin[0] <<endl;
      }
      t         =  cos[0] * UH(j,i) + conj(sin[0]) * UH(j,i+1);
      UH(j,i+1) = -sin[0] * UH(j,i) + cos[0]       * UH(j,i+1);
      UH(j,i) = t;
      if(ah_debug) {
	//if(abs(UH(j,i+1)) < 1e-10) UH(j,i+1) = 0.0;
	//if(abs(UH(j,i  )) < 1e-10) UH(j,i  ) = 0.0;
	cout<<"post h("<<j<<","<<i+1<<")="<< UH(j,i+1)<<endl;
	cout<<"post h("<<j<<","<<i<<")="<< UH(j,i)<<endl;
      }
    }

    //%-----------------------------------------------------%
    //| Accumulate the rotation in the matrix Q;  Q <- Q*G' |
    //%-----------------------------------------------------%
    // do 70
    if(ah_debug) cout << "min70 = " << std::min(i+1 + shift_num+1, nKr) << endl;
    for(int j = 0; j<std::min(i+1 + shift_num+1, nKr); j++) {
      if(ah_debug) {
	cout<<"pre q("<<j<<","<<i+1<<")="<< Q(j,i+1)<<endl;
	cout<<"pre q("<<j<<","<<i<<")="<< Q(j,i)<<endl;
	cout<<"cos=" << cos[0] << " sin = " << sin[0] <<endl;
      }
      t        =  cos[0] * Q(j,i) + conj(sin[0]) * Q(j,i+1);
      Q(j,i+1) = -sin[0] * Q(j,i) + cos[0]       * Q(j,i+1);
      Q(j,i) = t;
      if(ah_debug) {
	cout<<"post q("<<j<<","<<i+1<<")="<< Q(j,i+1)<<endl;
	cout<<"post q("<<j<<","<<i<<")="<< Q(j,i)<<endl;
      }
    }
    
    if(i < iend-2) {
      f = UH(i+1,i);
      g = UH(i+2,i);
    }
  }	  
}


void givensQRUpperHess(Eigen::MatrixXcd &UH, Eigen::MatrixXcd &Q, int nKr,
		       int shifts, int shift_num, int step_start, Complex shift, int iter){
  
  //https://cug.org/5-publications/proceedings_attendee_lists/1997CD/S96PROC/345_349.PDF
  // This code was put in place to deal with the difference between IEEE and CRAY
  // double prec formats. Now that we use IEEE almost exclusivley, we use the IEEE
  // starndards
  double unfl = DBL_MIN;
  double ulp = DBL_EPSILON;
  double smlnum = unfl*(mat_size/ulp);
  
  //%----------------------------------------%
  //| Check for splitting and deflation. Use |
  //| a standard test as in the QR algorithm |
  //| REFERENCE: LAPACK subroutine zlahqr    |
  //%----------------------------------------%

  int istart = 0;
  int iend = -1;

  bool g_debug = true;
  
  //znapps.f line 281
  // do 30 loop
  for(int i=istart; i<nKr-1; i++) {
    
    double tst1 = abs(UH(i,i).real()) + abs(UH(i,i).imag()) + abs(UH(i+1,i+1).real()) + abs(UH(i+1,i+1).imag());
    
    if( tst1 == 0 ) {
      cout << " *** TST1 hit "<< endl; 
      tst1 = zlanhs(UH);
    }

    if(g_debug) {
      //cout << "TEST at Iter = " << iter << " loop = " << i << " shift = " << shift_num << endl;
      //cout << tst1 << " " << abs(UH(i+1,i).real()) << " " << std::max(100*ulp*tst1, smlnum) << endl;
    }
    if (abs(UH(i+1,i).real()) <= std::max(100*ulp*tst1, smlnum)) {
      if(g_debug) cout << "UH split at " << i << " shift = " << shift_num << endl;
      iend = i+1;
      if(g_debug) cout << "istart = " << istart << " iend = " << iend << endl;
      UH(i+1,i) = 0.0;
      if(istart == iend){
	//if(istart == i){
	
	//%------------------------------------------------%
	//| No reason to apply a shift to block of order 1 |
	//| or if the current block starts after the point |
	//| of compression since we'll discard this stuff  |
	//%------------------------------------------------%    
	
	if(g_debug) cout << " No need for single block rotation at " << i << endl;
      } else if (istart > step_start) {
	if(g_debug) cout << " block rotation beyond " << step_start << endl;
      } else if( istart <= step_start) {
	applyShift(UH, Q, istart, iend, nKr, shift, shift_num, iter);
      }
      istart = iend + 1;
    }
  }

  iend = nKr;
  if(g_debug) {
    cout << "At End istart = " << istart << " iend = " << iend << endl;
    if(istart == iend) cout << " No need for single block rotation at " << istart << endl;
  }
  // If we finish the i loop with a istart less that step_start, we must
  // do a final set of shifts
  if(istart <= step_start && istart != iend) {
    //perform final block compression
    applyShift(UH, Q, istart, iend, nKr, shift, shift_num, iter);
  }
  
  //%---------------------------------------------------%
  //| Perform a similarity transformation that makes    |
  //| sure that the compressed H will have non-negative |
  //| real subdiagonal elements.                        |
  //%---------------------------------------------------%
  
  if( shift_num == shifts-1 ) {
    //do 120
    for(int j=0; j<step_start; j++) {
      if (UH(j+1,j).real() < 0.0 || UH(j+1,j).imag() != 0.0 ) {
	Complex t = UH(j+1,j) / dlapy2(UH(j+1,j).real(), UH(j+1,j).imag());	
	for(int i=0; i<nKr-j; i++) UH(j+1,i) *= conj(t);
	for(int i=0; i<std::min(j+1+2, nKr); i++) UH(i,j+1) *= t;
	for(int i=0; i<std::min(j+1+shifts+1,nKr); i++) Q(i,j+1) *= t;
	UH(j+1,j).imag(0.0);
      }
    }
    
    //do 130
    for(int i=0; i<step_start; i++) {

      //%--------------------------------------------%
      //| Final check for splitting and deflation.   |
      //| Use a standard test as in the QR algorithm |
      //| REFERENCE: LAPACK subroutine zlahqr.       |
      //| Note: Since the subdiagonals of the        |
      //| compressed H are nonnegative real numbers, |
      //| we take advantage of this.                 |
      //%--------------------------------------------%
      
      double tst1 = abs(UH(i,i).real()) + abs(UH(i,i).imag()) + abs(UH(i+1,i+1).real()) + abs(UH(i+1,i+1).imag());
      if( tst1 == 0 ) {
	cout << " ********* TST1 hit ********** "<< endl;
	tst1 = zlanhs(UH);
      }
      if (abs(UH(i+1,i).real()) <= std::max(100*ulp*tst1, smlnum)) {
	UH(i+1,i) = 0.0;
	if(g_debug) cout << "Zero out UH("<<i+1<<","<<i<<") by hand"<<endl;;
      }
    }
  }
}

	    

//SUBROUTINE ZLAHQR( WANTT, WANTZ, N, ILO, IHI, H, LDH, W, ILOZ, IHIZ, Z, LDZ, INFO )
//zlahqr(true, true, n, 0, n, upperHessTemp, ldh, ritz, 0, n, Q, ldq);

void zlahqr(bool WANTT, bool WANTZ, int N, int ILO, int IHI,
	    Eigen::MatrixXcd &UH, int LDH, std::vector<Complex> ritz,
	    int ILOZ, int IHIZ, Eigen::MatrixXcd &Q, int IDQ) {
  
  Complex cZero(0.0,0.0);
  Complex cOne(1.0,0.0);

  int I1, I2, ITN, ITS, k, l, m, NH, NZ;
  double H10, H21, RTEMP, S, SMLNUM, T2, TST1, ULP, UNFL;
  Complex CDUM, H11, H11S, H22, SUM, T, T1, TEMP, U, V2, X, Y;

  std::vector<double> rwork(1,0.0);
  std::vector<Complex> V(2,cZero);
  
  //Quick return if possible

  if( N == 0 ) return;
  if( ILO == IHI ) {
    ritz[ILO] = UH(ILO, ILO);
    return;
  }
  NH = IHI - ILO;   //Index change NH = IHI - ILO + 1 
  NZ = IHIZ - ILOZ; //Index change NZ = IHIZ - ILOZ + 1

  //Set machine-dependent constants for the stopping criterion.
  //If norm(H) <= sqrt(OVFL), overflow should not occur.  
  ULP = DBL_EPSILON;
  UNFL = DBL_MIN;
  SMLNUM = UNFL / ULP;

  //I1 and I2 are the indices of the first row and last column of H
  //to which transformations must be applied. If eigenvalues only are
  //being computed, I1 and I2 are set inside the main loop.  
  if( WANTT ) {
    I1 = 0; //Index change I1 = 1
    I2 = N;
  }
  
  //ITN is the total number of QR iterations allowed.  
  ITN = 30*NH;
  
  //The main loop begins here. I is the loop index and decreases from
  //IHI to ILO in steps of 1. Each iteration of the loop works
  //with the active submatrix in rows and columns L to I.
  //Eigenvalues I+1 to IHI have already converged. Either L = ILO, or
  //H(L,L-1) is negligible so that the matrix splits.
  int i = IHI-1; //Index change I = IHI
                 // Careful with this one. A matrix in f77 with
                 // M(IHI,IHI) is valid. 
  //10 continue
  //cout << "ILO = " << ILO << " IHI = " << IHI << " i = " << i << endl;
  while(i >= ILO) {
    
    //for(int w=0; w<LDH; w++) cout << "zlahqr ritz = " << ritz[i] << endl;
    
    //Perform QR iterations on rows and columns ILO to I until a
    //submatrix of order 1 splits off at the bottom because a
    //subdiagonal element has become negligible.    
    l = ILO;
    // do 110
    bool lgei = false;
    for(ITS = 0; ITS < ITN && !lgei; ITS++) {

      cout << "Start iteration " << ITS << " ILO = " << ILO << endl;
      
      //Look for a single small subdiagonal element.
      for(k = i; k >= l + 1; k--) {
	TST1 = cabs1( UH(k-1, k-1) ) + cabs1( UH(k, k));
	if(TST1 == 0.0) {
	  //cout << " *** TST1 hit "<< endl; 
	  TST1 = zlanhs(UH);
	}
	if(abs(UH(k, k-1).real()) <= std::max(ULP*TST1, SMLNUM)) {
	  //cout << " goto 30 hit "<< endl; 
	  continue;
	}
      }

      l = k;
      //cout << "ONE k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
      if (l > ILO) {
	//UH(L,L-1) is negligible
	UH(l, l-1) = cZero;
      }

      //Exit from loop if a submatrix of order 1 has split off.      
      if( l >= i ) {
	cout << "SM split at l=" << l << " i=" << i; 
	//UH(I,I-1) is negligible: one eigenvalue has converged.	
	ritz[i] = UH(i,i);
	  
	//Decrement number of remaining iterations, and return to start of
	//the main loop with new value of I.	  
	ITN = ITN - ITS;
	i = l - 1;
	// return to start of 10 continue 
	lgei = true;
      }
      
      //cout << "l="<<l<<" k="<<k<<endl;
      //Now the active submatrix is in rows and columns L to I. If
      //eigenvalues only are being computed, only the active submatrix
      //need be transformed.
      if(!WANTT) {
	I1 = l;
	I2 = i;
      }      
      if(ITS == 9 || ITS == 19 ) {
	//Exceptional shift.	
	T = abs(UH(i,i-1).real()) + abs(UH(i-1,i-2).real());
      } else {
	//Wilkinson's shift.	
	T = UH(i,i);
	U = UH(i-1,i) * UH(i,i-1).real();
	if(U != cZero) {	  
	  X = 0.5*(UH(i-1,i-1) - T);
	  Y = sqrt(X*X+U);
	  if( (X.real() * Y.real() + X.imag() * Y.imag()) < 0.0 ) Y = -Y;
	  T -= U/( X+Y );
	}
      }

      //Look for two consecutive small subdiagonal elements.
      bool goto50 = false;
      for(m = i - 1; m >= l + 1 && !goto50; m--) {

	//cout << "TWO k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
	
	//Determine the effect of starting the single-shift QR
	//iteration at row M, and see if this would make H(M,M-1)
	//negligible.

	H11 = UH(m, m);
	H22 = UH(m+1, m+1);
	H11S = H11 - T;
	//H21 = UH( M+1, M );
	H21 = UH(m+1, m).real(); //DMH guess this is what f77 means?? 
	S = cabs1( H11S ) + abs( H21 );
	H11S = H11S / S;
	H21 = H21 / S;
	V[0] = H11S;
	V[1] = H21;
	H10 = UH(m, m-1).real();//DMH guess this is what f77 means?? 
	TST1 = cabs1( H11S )*( cabs1( H11 ) + cabs1( H22 ) );
	if(abs( H10*H21 ) <= ULP*TST1 ) {
	  //cout << "goto50 hit" << endl;
	  goto50 = true;
	}
      }
      
      if(!goto50) {
	//cout << "!goto50 hit" << endl;
	//cout << "m="<<m<<" i="<<i<<" l="<<l<<endl;
	H11 = UH(l, l);
	H22 = UH(l+1, l+1);
	H11S = H11 - T;
	H21 = UH(l+1, l).real();//DMH guess this is what f77 means??
	S = cabs1( H11S ) + abs( H21 );
	H11S = H11S / S;
	H21 = H21 / S;
	V[0] = H11S;
	V[1] = H21;
      }

      //cout << "THREE k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
      
      //Single-shift QR step
      for(k = m; k < i; k++) {

	//cout << "FOUR k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
	
	//The first iteration of this loop determines a reflection G
	//from the vector V and applies it from left and right to H,
	//thus creating a nonzero bulge below the subdiagonal.
	
	//Each subsequent iteration determines a reflection G to
	//restore the Hessenberg form in the (K-1)th column, and thus
	//chases the bulge one step toward the bottom of the active
	//submatrix.
	
	//V[1] is always real before the call to ZLARFG, and hence
	//after the call T2 ( = T1*V[1] ) is also real.
	
	if(k > m) {
	  //CALL ZCOPY( 2, H( K, K-1 ), 1, V, 1 )
	  V[0] = UH(k, k-1);
	  V[1] = UH(k+1, k-1);
	}
	
	zlarfg(2, V[0], &V[1], 1, T1);
	if(k > m) {	    
	  UH(k, k-1) = V[0];
	  UH(k+1, k-1) = cZero;
	}
	V2 = V[1];
	Complex tempT2 = (T1*V2);
	T2 = tempT2.real();
	
	//Apply G from the left to transform the rows of the matrix
	//in columns K to I2.
	
	for (int j = k; j < I2; j++) {
	  SUM = conj(T1) * UH(k,j) + T2*UH(k+1,j);
	  UH(k,j) -= SUM;
	  UH(k+1,j) -= SUM*V2;
	}
	
	//Apply G from the right to transform the columns of the
	//matrix in rows I1 to min(K+2,I).
	
	for( int j = I1; j < std::min(k+2+1, i); j++) {
	  SUM = T1 * UH(j,k) + T2 * UH(j,k+1);
	  UH(j,k) -= SUM;
	  UH(j,k+1) -= SUM*conj(V2);
	}
	
	if( WANTZ ) {	    
	  //Accumulate transformations in the matrix Z	    
	  for (int j = ILOZ; j < IHIZ; j++) {
	    SUM = T1 * Q(j,k) + T2 * Q(j,k+1);
	    Q(j,k) -= SUM;
	    Q(j,k+1) -= SUM * conj(V2);
	  }
	}
	
	if( k == m && m > l ) {
	  //If the QR step was started at row M > L because two
	  //consecutive small subdiagonals were found, then extra
	  //scaling must be performed to ensure that H(M,M-1) remains
	  //real.
	  
	  TEMP = cOne - T1;
	  TEMP /= abs( TEMP );
	  UH(m+1, m) *= conj( TEMP );
	  if( m+2 <= i ) {
	    UH(m+1, m+1) *= TEMP;
	  }
	  for(int j = m; j < i; j++) { 
	    if( j != m+1 ) {
	      //if( I2 > j ) zscal( I2-j, TEMP, UH(j,j+1).data(), LDH );
	      if( I2 > j ) zscal( I2-j, TEMP, (UH.data() + j*LDH + j+1), LDH );
	      zscal( j-I1, conj( TEMP ), (UH.data() + I1*LDH + j), 1 );
	      if( WANTZ ) {
		zscal( NZ, conj( TEMP ), (Q.data() + ILOZ*LDH + j), 1 );
	      }
	    }
	  }
	}
      }

      //cout << "FIVE k="<<k<<" m="<<m<<" l="<<l<<" i="<<i<<endl;
      //Ensure that UH(I,I-1) is real.      
      TEMP = UH(i, i-1);
      if( TEMP.imag() != 0.0 ) {
	RTEMP = abs( TEMP );
	UH(i, i-1) = RTEMP;
	TEMP /= RTEMP;
	if( I2 > i ) zscal( I2-i, conj( TEMP ), (UH.data() + i*LDH+ i+1), LDH );
	zscal( i-I1, TEMP, (UH.data() + I1*LDH + i), 1 );
	if( WANTZ ) zscal( NZ, TEMP, (Q.data() + ILOZ*LDH + i), 1 );
      }
    }
  }
}
