require 'nn'
require 'optim'
require 'xlua'

local testing = require 'testing'


local train_network = function(cnn, training_set, validation_set)
  local criterion = nn.CrossEntropyCriterion()

  -- get the flattened parameters
  parameters, grad_parameters = cnn:getParameters()

  -- Use confusion matrix for error tracking
  classes = {}
  for i=1,43 do classes[i] = i end
  confusion = optim.ConfusionMatrix(classes)

  optimization_method = 'CG'

  local optimState
  if optimization_method == 'CG' then
    optimState = {
      maxIter = 20
    }
    optimMethod = optim.cg

  elseif optimization_method == 'LBFGS' then
    optimState = {
      learningRate = 0.01,
      maxIter = 2,
      nCorrection = 10
    }
    optimMethod = optim.lbfgs

  elseif optimization_method == 'SGD' then
    optimState = {
      learningRate = 0.1,
      weightDecay = 0,
      momentum = 0.5,
      learningRateDecay = 0.001
    }
    optimMethod = optim.sgd
  end
  
  nbr_epoch = 2
  batch_size = 100

  cnn:training()

  for epoch = 1, nbr_epoch do
    collectgarbage()
    print('Doing epoch ' .. epoch .. ' with batch of size ' .. batch_size)

    shuffle = torch.randperm(training_set:size())

    for t = 1, training_set:size(), batch_size do
      xlua.progress(t, training_set:size())
      -- generate batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batch_size-1,training_set:size()) do
        local input = training_set[shuffle[i]][1]
        local target = training_set[shuffle[i]][2]
        table.insert(inputs, input)
        table.insert(targets, target)
      end

      -- closure function to evaluate f(X) and df/dX
      local feval = function(x)
        if x ~= parameters then
          parameters:copy(x)
        end

        grad_parameters:zero()

        local f = 0

        for i = 1, #inputs do
          local output = cnn:forward(inputs[i])
          local err = criterion:forward(output, targets[i])
          f = f + err

          local df_do = criterion:backward(output, targets[i])
          cnn:backward(inputs[i], df_do)
        end

        grad_parameters:div(#inputs)
        f = f/#inputs

        return f, grad_parameters
      end

      optimMethod(feval, parameters, optimState)
    end

    testing.test_network(cnn, validation_set)
  end

end


return {
  train_network = train_network
}