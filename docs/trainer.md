This class train an SVM.

## svmsparse.Trainer(dim, lambda) ##

returns a new trainer for an SVM taking inputs of dimension dim, with a regularization parameter lambda.

Please see [Leon Bottou's page](http://leon.bottou.org/projects/sgd) for the loss equation and lambda definition.

## trainer:epoch(dataset) ##

Performs one SGD epoch, i.e. one pass over the training [dataset](https://github.com/DavidGrangier/svmsparse/blob/master/docs/dataset.md). 
Example,

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

## [res] trainer:evaluate(dataset) ##

returns a table containing averaged performance over dataset, where
res.loss (loss function), res.cost (loss + regularizer), res.err (error rate).

## [res] trainer:predict(dataset[, output]) ##

returns a torch.FloatTensor with the predictions over the dataset.
This tensor is a 1D tensor with one (float) prediction per example.
Optionally one can provide the output tensor to avoid memory allocation.

## [res] Trainer:weights() ##

returns the weights and bias of the SVM as a float tensor of dimension (dim + 1).
The bias is the last element.
