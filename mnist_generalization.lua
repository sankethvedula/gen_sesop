require "mnist"
require "nn"
require "cunn"
require "optim"


-- train and test data
train_data = torch.ones(60000,28,28)
test_data = torch.ones(10000,1)

network_layers = {512,512,512,512,512}

-- create a network
 model = nn.Sequential()
 model:add(nn.View(28*28))
 model:add(nn.Linear(28*28,network_layers[1]))
 model:add(nn.ReLU())
 model:add(nn.BatchNormalization(network_layers[1]))
 model:add(nn.Linear(network_layers[1],network_layers[2]))
 model:add(nn.ReLU())
 model:add(nn.BatchNormalization(network_layers[2]))
 model:add(nn.Linear(network_layers[2],network_layers[3]))
 model:add(nn.ReLU())
 model:add(nn.BatchNormalization(network_layers[3]))
 model:add(nn.Linear(network_layers[3],network_layers[4]))
 model:add(nn.ReLU())
 model:add(nn.BatchNormalization(network_layers[4]))
 model:add(nn.Linear(network_layers[4],network_layers[5]))
 model:add(nn.ReLU())
 model:add(nn.BatchNormalization(network_layers[2]))
 model:add(nn.Linear(network_layers[5],1))

 pred_out = model:forward(train_data)
 print(pred_out:size())


-- Training Error and Validation Error Log
training_log = optim.Logger('train_mnist_batch_'..batch_size..'.log')
validation_log = optim.Logger('validation_mnist_batch_'..batch_size..'.log')

training_log:setNames{'Training Loss'}
validation_log:setNames{'Validation Loss'}
