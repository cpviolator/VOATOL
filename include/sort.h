#pragma once

#include "voatol_internal.h"

void zsortc(int which, int n, std::vector<Complex> &x, std::vector<Complex> &y);
// Overloaded version of zsortc to deal with real y array.
void zsortc(int which, int n, std::vector<Complex> &x, std::vector<double> &y);
// Overloaded version of zsortc to deal with real x array.
void zsortc(int which, int n, std::vector<double> &x, std::vector<Complex> &y);
// Overloaded version of zsortc to deal with real x and y array.
void zsortc(int which, int n, std::vector<double> &x, std::vector<double> &y);

