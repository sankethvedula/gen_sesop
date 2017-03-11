require "torch"
require "image"

inputs_table = torch.load("clean_table_of_patches.t7")
outputs_table = torch.load("noisy_table_of_patches.t7")

train_inputs = {}
validation_inputs = {}

train_outputs = {}
validation_outputs = {}

for i = 1,2000 do
    table.insert(train_inputs,inputs_table[i])
    table.insert(train_outputs,outputs_table[i])
end

for i = 2500,3000 do
  table.insert(validation_inputs,inputs_table[i])
  table.insert(validation_outputs,outputs_table[i])
end

torch.save("train_inputs.t7",train_inputs)
torch.save("train_outputs.t7",train_outputs)
torch.save("validation_inputs.t7",validation_inputs)
torch.save("validation_outputs.t7",validation_outputs)
