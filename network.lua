require 'nn'


local get_network = function()
  local cnn = nn.Sequential()

  -- TODO use one of the improved architectures

  cnn:add(nn.SpatialConvolutionMM(3,12,5,5,1,1,2))
  cnn:add(nn.SpatialSubSampling(12, 2, 2, 2, 2))
  cnn:add(nn.Tanh())
  -- cnn:add(nn.SpatialContrastiveNormalization(12, torch.Tensor(4,4):fill(1)))

  cnn:add(nn.SpatialConvolutionMM(12,48,5,5,1,1,2))
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


local get_network_multiscale = function()
  local cnn = nn.Sequential()

  -- TODO use one of the improved architectures

  cnn:add(nn.SpatialConvolutionMM(3,12,5,5,1,1,2))
  cnn:add(nn.SpatialSubSampling(12, 2, 2, 2, 2))
  cnn:add(nn.Tanh())
  -- cnn:add(nn.SpatialContrastiveNormalization(12, torch.Tensor(4,4):fill(1)))

  local branch = nn.Concat(1)
  local branch_1 = nn.Sequential()
  branch_1:add(nn.SpatialConvolutionMM(12,48,5,5,1,1,2))
  branch_1:add(nn.SpatialSubSampling(48, 2, 2, 2, 2))
  branch_1:add(nn.Tanh())
  local branch_2 = nn.SpatialSubSampling(12, 2, 2, 2, 2)

  branch:add(branch_1)
  branch:add(branch_2)

  cnn:add(branch)


  cnn:add(nn.Reshape(1,(48+12)*8*8))
  cnn:add(nn.Linear((48+12)*8*8, 100))
  cnn:add(nn.Tanh())
  cnn:add(nn.Linear(100, 100))
  cnn:add(nn.Tanh())
  cnn:add(nn.Linear(100, 43))

  return cnn
end


return {
  get_network = get_network,
  get_network_multiscale = get_network_multiscale
}