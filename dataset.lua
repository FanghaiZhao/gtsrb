local path = require 'pl.path'
local dir = require 'pl.dir'
local data = require 'pl.data'
local utils = require 'pl.utils'

require 'image'
local torch = require 'torch'

-- prepare either the train or test dataset
-- returns test_dataset               if test_set is true
-- returns train_dataset, val_dataset if test_set is false
function get_dataset(test_set)
  local train_dataset = {}
  train_dataset.nbr_elements = 0
  function train_dataset:size() return train_dataset.nbr_elements end

  local val_dataset = {}
  val_dataset.nbr_elements = 0
  function val_dataset:size() return val_dataset.nbr_elements end

  local test_dataset = {}
  test_dataset.nbr_elements = 0
  function test_dataset:size() return test_dataset.nbr_elements end

  local parent_path
  if test_set then 
    parent_path = './GTSRB/Final_Test/Images'
  else
    parent_path = './GTSRB/Final_Training/Images'
  end

  local image_directories = dir.getdirectories(parent_path)

  for image_dir_nbr, image_directory in ipairs(image_directories) do
    local csv_file_name = 'GT-' .. path.basename(image_directory) .. '.csv'
    local csv_file_path = path.join(image_directory, csv_file_name)

    local csv_content = data.read(csv_file_path)

    local filename_index = csv_content.fieldnames:index('Filename')
    local class_id_index = csv_content.fieldnames:index('ClassId')
    local x1_index = csv_content.fieldnames:index('Roi_X1')
    local x2_index = csv_content.fieldnames:index('Roi_X2')
    local y1_index = csv_content.fieldnames:index('Roi_Y1')
    local y2_index = csv_content.fieldnames:index('Roi_Y2')


    -- first pass to detect number of tracks for this class
    local track_for_validation

    if test_set then
      -- no validation when working on the test_set
      track_for_validation = -1
    else
      local max_track_nbr = 0
      for image_index, image_metadata in ipairs(csv_content) do
        local track_nbr = tonumber(utils.split(image_metadata[filename_index], '_')[1])
        if track_nbr > max_track_nbr then
          max_track_nbr = track_nbr
        end
      end

      track_for_validation = torch.floor(torch.rand(1)*max_track_nbr) + 1 
    end

    for image_index, image_metadata in ipairs(csv_content) do
      local track_nbr = tonumber(utils.split(image_metadata[filename_index], '_')[1])
      local image_path = path.join(image_directory, image_metadata[filename_index])
      local image_data = torch.Tensor(image.load(image_path, 3, double))

      local x1 = image_metadata[x1_index]
      local x2 = image_metadata[x2_index]
      local y1 = image_metadata[y1_index]
      local y2 = image_metadata[y2_index]

      image_data = image.crop(image_data, x1, y1, x2, y2)
      image_data = image.scale(image_data, 32, 32)

      image_data = image.rgb2yuv(image_data)

      image_data = image_data - torch.mean(image_data)

      local label = torch.Tensor(1)
      label[1] = image_metadata[class_id_index]+1

      if test_set then
        test_dataset.nbr_elements = test_dataset.nbr_elements + 1
        test_dataset[test_dataset.nbr_elements] = {image_data, label}
      else
        if track_nbr == track_for_validation[1] then
          val_dataset.nbr_elements = val_dataset.nbr_elements + 1
          val_dataset[val_dataset.nbr_elements] = {image_data, label}
        else
          train_dataset.nbr_elements = train_dataset.nbr_elements + 1
          train_dataset[train_dataset.nbr_elements] = {image_data, label}
        end
      end

    end

  end

  if test_set then
    return test_dataset
  else
    return train_dataset, val_dataset
  end
end