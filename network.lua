require 'nn'


local get_network = function()
  local cnn = nn.Sequential()

  -- TODO use one of the improved architectures

  cnn:add(nn.SpatialConvolution(3,12,5,5,1,1,2))
  cnn:add(nn.SpatialSubSampling(12, 2, 2, 2, 2))
  cnn:add(nn.Tanh())
  cnn:add(nn.SpatialConvolution(12,48,5,5,1,1,2))
  cnn:add(nn.SpatialSubSampling(48, 2, 2, 2, 2))
  cnn:add(nn.Tanh())


  cnn:add(nn.Reshape(1,48*8*8))
  cnn:add(nn.Linear(48*8*8, 100))
  cnn:add(nn.Tanh())
  cnn:add(nn.Linear(100, 100))
  cnn:add(nn.Tanh())
  cnn:add(nn.Linear(100, 43))

  return cnn
end

return {
  get_network = get_network
}