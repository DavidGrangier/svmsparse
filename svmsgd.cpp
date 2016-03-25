// -*- C++ -*-
// SVM with stochastic gradient
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

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "loss.h"
#include "svmsgd.h"

using namespace std;

// ---- Loss function

// Compile with -DLOSS=xxxx to define the loss function.
// Loss functions are defined in file loss.h)
#ifndef LOSS
# define LOSS LogLoss
#endif

// ---- Bias term

// Compile with -DBIAS=[1/0] to enable/disable the bias term.
// Compile with -DREGULARIZED_BIAS=1 to enable regularization on the bias term

#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif

// ---- Plain stochastic gradient descent

/// Constructor
SvmSgd::SvmSgd(int dim, double lambda, double eta0)
  : w(dim), wBias(0), lambda(lambda), eta0(eta0), wDivisor(1), t(0)
{
}

/// Renormalize the weights
void
SvmSgd::renorm()
{
  if (wDivisor != 1.0)
    {
      w.scale(1.0 / wDivisor);
      wDivisor = 1.0;
    }
}

/// Compute the norm of the weights
double
SvmSgd::wnorm()
{
  double norm = dot(w,w) / wDivisor / wDivisor;
#if REGULARIZED_BIAS
  norm += wBias * wBias;
#endif
  return norm;
}

/// Compute the output for one example.
double
SvmSgd::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
  double s = dot(w,x) / wDivisor + wBias;
  if (ploss)
    *ploss += LOSS::loss(s, y);
  if (pnerr)
    *pnerr += (s * y <= 0) ? 1 : 0;
  return s;
}

/// Perform one iteration of the SGD algorithm with specified gains
void
SvmSgd::trainOne(const SVector &x, double y, double eta)
{
  double s = dot(w,x) / wDivisor + wBias;
  // update for regularization term
  wDivisor = wDivisor / (1 - eta * lambda);
  if (wDivisor > 1e5) renorm();
  // update for loss term
  double d = LOSS::dloss(s, y);
  if (d != 0)
    w.add(x, eta * d * wDivisor);
  // same for the bias
#if BIAS
  double etab = eta * 0.01;
#if REGULARIZED_BIAS
  wBias *= (1 - etab * lambda);
#endif
  wBias += etab * d;
#endif
}

/// Perform a training epoch
void
SvmSgd::train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
#if VERBOSE
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
#endif
  assert(imin <= imax);
  assert(eta0 > 0);
  for (int i=imin; i<=imax; i++)
    {
      double eta = eta0 / (1 + lambda * eta0 * t);
      trainOne(xp.at(i), yp.at(i), eta);
      t += 1;
    }
#if VERBOSE
  cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
  cout << " wBias=" << wBias;
#endif
  cout << endl;
#endif
}

/// Perform a test pass
void
SvmSgd::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  double nerr = 0;
  double loss = 0;
  for (int i=imin; i<=imax; i++)
    testOne(xp.at(i), yp.at(i), &loss, &nerr);
  nerr = nerr / (imax - imin + 1);
  loss = loss / (imax - imin + 1);
  double cost = loss + 0.5 * lambda * wnorm();
  cout << prefix
       << "Loss=" << setprecision(12) << loss
       << " Cost=" << setprecision(12) << cost
       << " Misclassification=" << setprecision(4) << 100 * nerr << "%."
       << endl;
}

/// Perform one epoch with fixed eta and return cost
double
SvmSgd::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
  SvmSgd clone(*this); // take a copy of the current state
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    clone.trainOne(xp.at(i), yp.at(i), eta);
  double loss = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    clone.testOne(xp.at(i), yp.at(i), &loss, 0);
  loss = loss / (imax - imin + 1);
  cost = loss + 0.5 * lambda * clone.wnorm();
#if VERBOSE
  cout << "Trying eta=" << eta << " yields cost " << cost << endl;
#endif
  return cost;
}

void
SvmSgd::determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
{
  const double factor = 2.0;
  double loEta = 1;
  double loCost = evaluateEta(imin, imax, xp, yp, loEta);
  double hiEta = loEta * factor;
  double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
  if (loCost < hiCost)
    while (loCost < hiCost)
      {
        hiEta = loEta;
        hiCost = loCost;
        loEta = hiEta / factor;
        loCost = evaluateEta(imin, imax, xp, yp, loEta);
      }
  else if (hiCost < loCost)
    while (hiCost < loCost)
      {
        loEta = hiEta;
        loCost = hiCost;
        hiEta = loEta * factor;
        hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
      }
  eta0 = loEta;
#if VERBOSE
  cout << "# Using eta0=" << eta0 << endl;
#endif
}

FVector
SvmSgd::getWeights()
{
  return w;
}

double
SvmSgd::getBias()
{
  return wBias;
}
