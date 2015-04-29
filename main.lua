require 'dataset'
require 'network'
require 'training'
require 'testing'

local path = require 'pl.path'
require 'torch'

print('Reading training dataset.')
local train_dataset, val_dataset = get_dataset(false)
print('Using ' .. train_dataset.nbr_elements .. ' training samples.')
print('Using ' .. val_dataset.nbr_elements .. ' validation samples.')
print('Reading testing dataset.')
local test_dataset = get_dataset(true)
print('Using ' .. test_dataset.nbr_elements .. ' testing samples.')

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
test_network(mlp, test_dataset)
