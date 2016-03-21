// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2016- David Grangier

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

#include "SvmSgd.h"
#include "assert.h"

using namespace std;

/// dataset
class Dataset {
  private:
   int pcount;
   int ncount;
   int dim;
  public:
   xvec_t xp;
   yvec_t yp;
  public:
    Dataset();
    void add(SVector x, double y);
    int getDim();
};

Dataset::Dataset()
{
  pcount = 0;
  ncount = 0;
  dim = -1;
}

void
Dataset::add(SVector x, double y)
{
  if (y != +1 && y != -1)
    assertfail("Label should be +1 or -1.");
  xp.push_back(x);
  yp.push_back(y);
  if (y > 0)
    pcount += 1;
  else
    ncount += 1;
  if (x.size() > dim)
    dim = x.size();
}

int
Dataset::getDim()
{
  return dim;
}


/// exposed c interface
extern "C" {

Dataset*
dataset_new()
{
  return new Dataset();
}

void
dataset_add(Dataset* d, int n, int* indices, float* values, double label)
{
  SVector x;
  x.clear();
  for (int i = 0; i < n; i++) {
    x.set(*indices++ - 1, *values++);
  }
  x.trim();
  d->add(x, label);
}

int
dataset_getdim(Dataset *d)
{
  return d->getDim();
}

void
dataset_free(Dataset* d)
{
  delete d;
}

}

/// trainer
class Trainer {
  public:
    Trainer(int dim, double lambda);
    void epoch(Dataset dataset);
    void eval(Dataset dataset, double *loss, double *cost, double *nerr);
    int getDim();
    void getWeights(float* weights);
  private:
    int dim;
    double lambda;
    bool needEta0;
    SvmSgd svm;
};

Trainer::Trainer(int dim, double lambda)
  : dim(dim), lambda(lambda), needEta0(true), svm(dim, lambda)
{
}

void
Trainer::epoch(Dataset d)
{
  int imin = 0;
  int imax = d.xp.size() - 1;
  if (needEta0) {
    int smin = 0;
    int smax = imin + min(1000, imax);
    svm.determineEta0(smin, smax, d.xp, d.yp);
    needEta0 = false;
  }
  svm.train(imin, imax, d.xp, d.yp);
}

void
Trainer::eval(Dataset d, double *loss, double *cost, double *nerr)
{
  *loss = 0;
  *nerr = 0;
  *cost = 0;
  int imin = 0;
  int imax = d.xp.size() - 1;
  for (int i = imin; i <= imax; i++)
    svm.testOne(d.xp.at(i), d.yp.at(i), loss, nerr);
  *nerr = *nerr / (imax - imin + 1);
  *loss = *loss / (imax - imin + 1);
  *cost = *loss + 0.5 * lambda * svm.wnorm();
}

int
Trainer::getDim()
{
  return dim;
}

void
Trainer::getWeights(float* weights)
{
  FVector w = svm.getWeights();
  for (int i = 0; i < dim; i++) {
    *weights++ = w.get(i);
  }
  *weights = svm.getBias();
}

// exposed c interface
extern "C" {

Trainer*
trainer_new(int dim, double lambda)
{
  return new Trainer(dim, lambda);
}

void trainer_epoch(Trainer* t, Dataset* d)
{
  t->epoch(*d);
}

void
trainer_evaluate(Trainer* t, Dataset* d, double *loss, double *cost, double *nerr)
{
  t->eval(*d, loss, cost, nerr);
}

int trainer_weightdim(Trainer* t)
{
  return t->getDim() + 1;
}

void
trainer_getweights(Trainer* t, float* weights)
{
  t->getWeights(weights);
}

void
trainer_free(Trainer* t)
{
  delete t;
}

}
