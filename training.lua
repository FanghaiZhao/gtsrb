require 'nn'


function train_network(cnn, dataset)
  local criterion = nn.CrossEntropyCriterion()
  local trainer = nn.StochasticGradient(cnn, criterion)

  trainer.learningRate = 0.01
  trainer.maxIteration = 10 -- 25 is the default value
  trainer:train(dataset)

end