
local get_features = function(cnn, index_conv, input)
  
  local index = 0
  -- make a forward pass
  cnn:forward(input)

  local output = cnn.modules[index_conv].output
  local size = output:size(1) * output:size(2) * output:size(3)

  return output:reshape(1, size)
end


local all_filled = function(table, size)
  local i
  for i=1,size do
    if not table[i] then
      return false
    end
  end

  return true
end


local select_references = function(cnn, index_conv, dataset)
  
  local references = {}
  local index = 1
  while not all_filled(references, 43) do
    sample = dataset[index]
    if not references[sample[2][1]] then
      references[sample[2][1]] = get_features(cnn, index_conv, sample[1])
    end
    index = index + 1
  end

  return references
end


local distance_function = function(tensor1, tensor2)

  local diff = tensor1 - tensor2
  return torch.sqrt(diff:transpose(1,2):dot(diff))
end

local classify_intput = function(cnn, index_conv, references, sample)
  local sample_features = get_features(cnn, index_conv, sample)

  local best_choice = 0
  local best_distance = 1/0
  local dist
  for label, ref_features in ipairs(references) do
    dist = distance_function(ref_features, sample_features)
    if dist < best_distance then
      best_distance = dist
      best_choice = label
    end
  end
  return best_choice
end


local test_matching = function(cnn, index_conv, dataset)

  local references = select_references(cnn, index_conv, dataset)
  
  local nbr_elements = 0
  local nbr_false = 0
  local prediction

  for index, sample in ipairs(dataset) do
    prediction = classify_intput(cnn, index_conv, references, sample[1])
    if prediction ~= sample[2][1] then
      nbr_false = nbr_false + 1
    end
    nbr_elements = nbr_elements + 1
  end

  print('Error rate using matching on the given set is: ' .. nbr_false/nbr_elements .. '.')
end

return {
  test_matching = test_matching
}