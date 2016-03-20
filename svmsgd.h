// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2016- David Grangier
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#ifndef SVMSGD_H
#define SVMSGD_H

#include <vector>
#include "vectors.h"

typedef std::vector<SVector> xvec_t;
typedef std::vector<double>  yvec_t;

class SvmSgd
{
public:
  SvmSgd(int dim, double lambda, double eta0  = 0);
  void renorm();
  double wnorm();
  double testOne(const SVector &x, double y, double *ploss, double *pnerr);
  void trainOne(const SVector &x, double y, double eta);
public:
  void train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
public:
  double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
  void determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y);
  FVector getWeights();
  double getBias();
private:
  FVector w;
  double  wBias;
  double  lambda;
  double  eta0;
  double  wDivisor;
  double  t;
};


#endif
