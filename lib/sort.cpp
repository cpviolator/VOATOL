#include "sort.h"

void zsortc(int which, int n, std::vector<Complex> &x, std::vector<Complex> &y) {
  
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
