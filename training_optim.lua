require 'nn'
require 'optim'
require 'xlua'


function train_network(cnn, dataset)
  local criterion = nn.CrossEntropyCriterion()

  -- get the flattened parameters
  parameters, grad_parameters = cnn:getParameters()

  -- Use confusion matrix for error tracking
  classes = {}
  for i=1,43 do classes[i] = i end
  confusion = optim.ConfusionMatrix(classes)

  optimization_method = 'SGD'

  local optimState
  if optimization_method == 'CG' then
    optimState = {
      maxIter = 2
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
      learningRate = 0.01,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 0
    }
    optimMethod = optim.sgd
  end
  
  nbr_epoch = 10
  batch_size = 100

  cnn:training()

  for epoch = 1, nbr_epoch do

    print('Doing epoch ' .. epoch .. ' with batch of size ' .. batch_size)

    shuffle = torch.randperm(dataset:size())

    for t = 1, dataset:size(), batch_size do
      xlua.progress(t, dataset:size())
      -- generate batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batch_size-1,dataset:size()) do
        local input = dataset[shuffle[i]][1]
        local target = dataset[shuffle[i]][2]
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
  end

end
