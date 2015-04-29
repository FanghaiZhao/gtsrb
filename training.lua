require 'nn'


function train_network(mlp, dataset)
  local criterion = nn.CrossEntropyCriterion()
  local trainer = nn.StochasticGradient(mlp, criterion)

  trainer.learningRate = 0.01
  trainer.maxIteration = 25 -- 25 is the default value
  trainer:train(dataset)

end