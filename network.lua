require 'nn'


function get_network()
  local mlp = nn.Sequential()

  -- TODO use one of the improved architectures

  --mlp:add(nn.SpatialConvolution(3,10,5,5,1,1,2))
  mlp:add(nn.Reshape(1,3072))
  mlp:add(nn.Linear(3072, 100))
  mlp:add(nn.Tanh())
  mlp:add(nn.Linear(100, 100))
  mlp:add(nn.Tanh())
  mlp:add(nn.Linear(100, 43))

  return mlp
end