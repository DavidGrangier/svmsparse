# SVM Sparse for Torch #

This is a Torch interface to Leon Bottou's fast log-loss sparse linear SVM (logistic regression).
Currently only the plain sgd version (no average SGD) is interfaced.

Please see [Leon Bottou's page](http://leon.bottou.org/projects/sgd) for the original project.

## Requirement ##

Torch7 (http://github.com/torch/torch7)

## Installation ##

luarocks install https://raw.githubusercontent.com/DavidGrangier/svmsparse/master/rocks/svmsparse-scm-1.rockspec

## Usage ##

Create a training set from sparse vectors, triplets of (indices, values, labels)
where indices: IntTensor, values: FloatTensor, labels -1.0 or 1.0 . In the lua tradition, the indices of sparse vectors starts at 1.

```lua
require 'torch'
local svmsparse = require 'svmsparse'

-- create indices[i], values[i], labels[i]...
-- data should be shuffled, L2-normalized by your program.

local trainset = svmsparse.Dataset()
for i = 1, n do
  trainset:add(indices[i], values[i], labels[i])
end
```

Then trainer a model for a few epoch with
```lua
local nepoch = 1
local lambda = 1e-5
local trainer = svmsparse.Trainer(trainset:dim(), lambda)
for i = 1, nepoch do
  trainer:epoch(trainset)
  local eval = trainer:evaluate(trainset)
  print(string.format('loss = %.3f cost = %.3f err = %.3f', eval.loss, eval.cost, eval.err))
end
```

Finally, you can read the trained weights + bias as a FloatTensor,
```lua
local w = trainer:weights()
-- do what you want with the weights...
```

See example.lua for a running toy example.

## One Vs All ##

For one versus all multiclass setting, it is often usefull to keep the same training set and replace the labels, 

```lua
local n = trainset:size()
for i = 1, n do
  trainset:relabel(i, newlabels[i])
end
```

## Documentation ##

[svmsparse.Dataset](https://github.com/DavidGrangier/svmsparse/blob/master/docs/dataset.md)
[svmsparse.Trainer](https://github.com/DavidGrangier/svmsparse/blob/master/docs/trainer.md)
