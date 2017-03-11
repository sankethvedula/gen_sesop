require "torch"
require "image"
require "nn"

net = nn.Sequential()
net:add(nn.JoinTable(1))

name_of_file = "validation_outputs"

clean_table = torch.load(name_of_file..".t7")

joined = net:forward(clean_table)
print(joined:size())
out_file_name = name_of_file.."_tensor"
torch.save(out_file_name..".t7",joined)
