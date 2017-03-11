require "torch"
require "image"
require "nn"


folder_name = "./noisy_data/"
net = nn.Sequential()
net:add(nn.View(256,256))

table_of_tensors = {}
for file_number = 1,3000 do
  print(file_number)
  image1 = image.load(folder_name..file_number..".jpg",1,"byte")
  image1 = image1:double()

  --print(image1:size())

  if image1:size(1) == 1 then
      image1 = image1:reshape(image1:size(2),image1:size(3))
  else
    image1 = image1
  end

  table.insert(table_of_tensors,image1)

end

torch.save("noisy_images.t7",table_of_tensors)
