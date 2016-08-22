require 'torch'

local svmsparse = require 'svmsparse'

-- lua inference
local svm = function(w, x)
  local s = w[w:size(1)]
  local idx, val = unpack(x)
  for i = 1, idx:size(1) do
    s = s + w[ idx[i] ] * val[i]
  end
  return s
end

-- toy data
local data = {
 { torch.IntTensor{1,2,8}, torch.FloatTensor({1.0,1.0,1.0}), 1.0},
 { torch.IntTensor{1,4,9}, torch.FloatTensor({1.0,1.0,1.0}), 1.0}
}
local dataset = svmsparse.Dataset()
for _, x in ipairs(data) do
  local indices, value, label = unpack(x)
  dataset:add(indices, value, label)
end
assert(dataset:dim() == 9)

dataset:relabel(2, -1.0)
local size, pcount, ncount = dataset:size()
assert(size == 2 and pcount == 1 and ncount == 1)

-- training
local nepoch = 1
local lambda = 1e-5
local trainer = svmsparse.Trainer(dataset:dim(), lambda)
for i = 1, nepoch do
  trainer:epoch(dataset)
  local eval = trainer:evaluate(dataset)
  print(string.format('loss = %.3f cost = %.3f err = %.3f', eval.loss, eval.cost, eval.err))
end

-- test
local w = trainer:weights()
assert(w:size(1) - 1 == dataset:dim())
local m1 = svm(w, data[1]) * data[1][3]
local m2 = svm(w, data[2]) * data[2][3]
print(string.format('margin1 % 3.2f', m1))
print(string.format('margin1 % 3.2f', m2))
assert(m1 > 0)
assert(m2 > 0)
