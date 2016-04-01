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

#include <fstream>
#include <iostream>
#include "svmsgd.h"
#include "assert.h"

using namespace std;

typedef std::vector<int> ivec_t;

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
    void add(Dataset d, double pratio);
    void addclass(Dataset d, double y);
    void relabel(int i, double y);

    double getLabel(int i);
    SVector getExample(int i);

    int getDim();
    int getPCount();
    int getNCount();

    void load(istream &file);
    void save(ostream &file);
};

Dataset::Dataset()
  : pcount(0), ncount(0), dim(-1)
{
}

void
Dataset::add(SVector x, double y)
{
  if (y != +1 && y != -1)
    assertfail("Label should be +1 or -1.");
  xp.push_back(x);
  yp.push_back(y);
  if (y > 0)
    pcount++;
  else
    ncount++;
  if (x.size() > dim)
    dim = x.size();
}

void
Dataset::add(Dataset d, double pratio)
{
  int n = d.xp.size();
  ivec_t pos;
  for (int i=0; i<n; i++) {
    if (d.yp.at(i) > 0.0)
      pos.push_back(i);
  }

  int p = pos.size();
  int j = 0;
  for (int i=0; i<n; i++)
  {
    if (d.yp.at(i) < 0.0)
      add(d.xp.at(i), -1.0);
    if (j < pratio * (i+j))
    {
      int k = pos.at(j % p);
      add(d.xp.at(k), 1.0);
      j++;
    }
  }
}

void
Dataset::addclass(Dataset d, double y)
{
  int n = d.xp.size();
  for (int i=0; i<n; i++) {
    if (d.yp.at(i) == y)
      add(d.xp.at(i), y);
  }
}

void
Dataset::relabel(int i, double y)
{
  if (y != +1 && y != -1)
    assertfail("Label should be +1 or -1.");
  if (yp[i] > 0)
    pcount--;
  else
    ncount--;
  yp[i] = y;
  if (y > 0)
    pcount++;
  else
    ncount++;
}

SVector
Dataset::getExample(int i)
{
  if (i < 0 || i >= (int)xp.size())
    assertfail("getExample: index out of range");
  return xp.at(i);
}

double
Dataset::getLabel(int i)
{
  if (i < 0 || i >= (int)yp.size())
    assertfail("getLabel: index out of range");
  return yp.at(i);
}

int
Dataset::getDim()
{
  return dim;
}

int
Dataset::getPCount()
{
  return pcount;
}

int
Dataset::getNCount()
{
  return ncount;
}

void
Dataset::load(istream &file)
{
  int nexamples;
  file.read((char*)&nexamples, sizeof(int));
  SVector sv;
  double y;
  for(int i = 0; i < nexamples; i++) {
    sv.load(file);
    file.read((char*)&y, sizeof(double));
    add(sv, y);
  }
}

void
Dataset::save(ostream &file)
{
  int nexamples = pcount + ncount;
  file.write((char*)&nexamples, sizeof(int));
  double y;
  for(int i = 0; i < nexamples; i++) {
    xp.at(i).save(file);
    y = yp.at(i);
    file.write((char*)&y, sizeof(double));
  }
}

/// exposed c interface
extern "C" {

Dataset*
dataset_new()
{
  return new Dataset();
}

void
dataset_balance(Dataset *d, Dataset* src, double pratio) {
  d->add(*src, pratio);
}

void
dataset_oneclass(Dataset *d, Dataset* src, double label) {
  d->addclass(*src, label);
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

void
dataset_relabel(Dataset* d, int index, double label)
{
  d->relabel(index, label);
}

double
dataset_getlabel(Dataset* d, int i)
{
  return d->getLabel(i);
}

int
dataset_getnonzero(Dataset* d, int i) {
  return d->getExample(i).npairs();
}

void
dataset_getexample(Dataset* d, int i, int* indices, float* values) {
  SVector x = d->getExample(i);
  for(const SVector::Pair *p = x; p->i>=0; p++) {
    *indices++ = p->i + 1;
    *values++ = p->v;
  }
}

int
dataset_getdim(Dataset *d)
{
  return d->getDim();
}

int
dataset_getpcount(Dataset *d)
{
  return d->getPCount();
}

int
dataset_getncount(Dataset *d)
{
  return d->getNCount();
}

void
dataset_save(Dataset *d, const char* filename)
{
  ofstream file;
  file.open(filename);
  d->save(file);
  file.close();
}

void
dataset_load(Dataset *d, const char *filename)
{
  ifstream file;
  file.open(filename);
  d->load(file);
  file.close();
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
    yvec_t predict(Dataset dataset);
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

yvec_t
Trainer::predict(Dataset d)
{
  yvec_t pred;
  int n = d.xp.size();
  for (int i = 0; i < n; i++)
    pred.push_back( svm.testOne(d.xp.at(i), 0.0, NULL, NULL) );
  return pred;
}

int
Trainer::getDim()
{
  return dim;
}

void
Trainer::getWeights(float* weights)
{
  svm.renorm();
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

void
trainer_predict(Trainer* t, Dataset* d, float *pred) {
  yvec_t p = t->predict(*d);
  int n = p.size();
  for(int i=0; i<n; i++) {
    *pred++ = (float) p.at(i);
  }
}

int
trainer_weightdim(Trainer* t)
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
