# Dataset #

The dataset class allows to manipulate data to feed the [svmsparse.Trainer](https://github.com/DavidGrangier/svmsparse/blob/master/docs/trainer.md) class, which finds the SVM weights through SGD optimization.

## Construction ##

### [res] svmsparse.Dataset() ###

returns a new empty dataset. E.g. 
```lua
> svmsparse = require 'svmsparse'
> dataset = svmsparse.Dataset()
```

### [res] dataset:oneclass(label) ###

return a new dataset with all the examples whose class is label. 

### [res] dataset:balance(pratio) ###

return a new dataset such that the fraction of positive if pratio.
It does so by upsampling the positives and assumes the positive class is rarer than the negative class.

## Modifying ##

### dataset:add(indices, values, label) ###

add an example. The example sparse vector is encoded as a set of indices and values where only non-zero entries are required. 
Indices is a 1D torch.IntTensor where indices ranges between 1 and dim, while values is a torch.FloatTensor. Label is either 
1 or -1.

E.g.
 ```lua
> svmsparse = require 'svmsparse'
> dataset = svmsparse.Dataset()
> dataset:add(torch.IntTensor({1,4,7}), torch.FloatTensor({2.0, 1.2, 0.5}), 1.0)
```

### dataset:relabel(i, label) ###

change the class of the i th example to label. This is particularly usefull for one versus all training.

## Querying ##

### [dim] dataset:dim() ###

return the input dimension (largest non zero indices of the inputs).

### [count, pcount, ncount] dataset:size() ###

return the number of examples, as well as the number of positives and negatives.

## [indices, values, label] dataset:getExample(i, [indices, values]) ##

retrieve the i th example of the dataset. It returns (indices, values, label).
Optionally, one can provide a torch.IntTensor for indices and a torch.FloatTensor for values to avoid memory allocation.

## Load / Save Methods ##

### dataset:save(filename) ###

save all the examples into filename.

### dataset:load(filename) ###

add the examples stored in filename into dataset.
