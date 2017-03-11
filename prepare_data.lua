require "nn"
require "image"
require "lfs"

folder_name = "./sub_data/"

-- Add noise to the images

-- Iterate through all the images

for file_name =1,4000 do
  print(file_name)
  current_file_name = folder_name..file_name..".jpg"
  -- read the file_number
  image1 = image.load(current_file_name,1,"byte")
  image1 = image1:double()

  print(image1:size())
  if image1:size(1) == 1 then
    noise = torch.randn(image1:size(2),image1:size(3))
  else
    noise = torch.randn(image1:size(1),image1:size(2))
  end

  noise = noise:mul(math.sqrt(0.5*10))
  noisy_image = image1 + noise
  image.save("./noisy_data/"..file_name..".jpg",noisy_image)
end
