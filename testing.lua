require 'torch'


local test_network = function(cnn, dataset)

  local nbr_elements = 0
  local nbr_false = 0

  for sample_index, sample in ipairs(dataset) do
    score = cnn:forward(sample[1])

    local max_values, max_indices = torch.max(score,2)

    -- TODO handle case where size of max_indices is > 1

    if max_indices[1][1] ~= sample[2][1] then
      nbr_false = nbr_false + 1
      -- print(max_indices[1][1] .. ' ~= ' .. sample[2][1])
    end
    nbr_elements = nbr_elements + 1

  end

  print('Error rate on the given set is: ' .. nbr_false/nbr_elements .. '.')

end


return {
  test_network = test_network
}