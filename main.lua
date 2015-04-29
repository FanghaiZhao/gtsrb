require 'dataset'
require 'network'
require 'training'
require 'testing'

local path = require 'pl.path'
require 'torch'

print('Reading dataset.')
local train_dataset, val_dataset = get_dataset()
print('Using ' .. train_dataset.nbr_elements .. ' training samples.')
print('Using ' .. val_dataset.nbr_elements .. ' validation samples.')

local mlp

if path.exists('model.bin') then
  print('Using pretrained network.')
  mlp = torch.load('model.bin')
else
  mlp = get_network()
  print('Training network.')
  train_network(mlp, train_dataset)

  print('Saving network.')
  torch.save('model.bin', mlp)
end


print('Testing network.')
test_network(mlp, val_dataset)
