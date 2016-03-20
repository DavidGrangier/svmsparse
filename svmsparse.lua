-- -*- lua -*-
-- SVM with stochastic gradient
-- Copyright (C) 2016- David Grangier

-- This program is free software; you can redistribute it and/or
-- modify it under the terms of the GNU Lesser General Public
-- License as published by the Free Software Foundation; either
-- version 2.1 of the License, or (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with this program; if not, write to the Free Software
-- Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

require 'torch'
local ffi = require 'ffi'

ffi.cdef[[

typedef struct Dataset_ Dataset;
Dataset* dataset_new();
void dataset_add(Dataset* d, int n, int* indices, float* values, double label);
int dataset_getdim(Dataset *d);
void dataset_free(Dataset* d);

typedef struct Trainer_ Trainer;
Trainer* trainer_new(int dim, double lambda);
void trainer_epoch(Trainer* t, Dataset* d);
void trainer_evaluate(
  Trainer* t, Dataset* d, double *loss, double *cost, double *nerr);
int trainer_weightdim(Trainer* t);
void trainer_getweights(Trainer* t, float* weights);
void trainer_free(Trainer* t);

]]

--local C = ffi.load('svmsparse')
local C = ffi.load(package.searchpath('libsvmsparse', package.cpath))

--- ---
local Dataset = {}

function Dataset.new()
  local self = C.dataset_new()
  self = ffi.cast('Dataset&', self)
  ffi.gc(self, C.dataset_free)
  return self
end

function Dataset:add(indices, values, label)
  local n = indices:size(1)
  local i, v = indices:data(), values:data()
  C.dataset_add(self, n, i, v, label)
end

Dataset.dim = C.dataset_getdim;

ffi.metatype('Dataset', {__index = Dataset})
local Dataset_ctor = {}
setmetatable(Dataset_ctor, { __call = Dataset.new })

--- ---
local Trainer = {}

function Trainer.new(_, dim, lambda)
  lambda = lambda or 1e-5;
  local self = C.trainer_new(dim, lambda)
  self = ffi.cast('Trainer&', self)
  ffi.gc(self, C.trainer_free)
  return self
end

Trainer.epoch = C.trainer_epoch

function Trainer:evaluate(dataset)
  local res = ffi.new('double[?]', 3)
  C.trainer_evaluate(self, dataset, res, res+1, res+2)
  return {
    loss = res[0],
    cost = res[1],
    err = res[2],
  }
end

function Trainer:weights()
  local d = C.trainer_weightdim(self)
  local w = torch.FloatTensor(d)
  C.trainer_getweights(self, w:data())
  return w
end

ffi.metatype('Trainer', {__index = Trainer})
local Trainer_ctor = {}
setmetatable(Trainer_ctor, { __call = Trainer.new })

return {
  Dataset = Dataset_ctor,
  Trainer = Trainer_ctor,
}
