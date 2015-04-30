require 'nn'


local train_network = function(cnn, dataset)
  local criterion = nn.CrossEntropyCriterion()
  local trainer = nn.StochasticGradient(cnn, criterion)

  trainer.learningRate = 0.01
  trainer.maxIteration = 10 -- 25 is the default value
  trainer:train(dataset)

end

return {
  train_network = train_network
}