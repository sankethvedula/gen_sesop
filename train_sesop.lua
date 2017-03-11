require "cunn"
require "nn"
require "image"
require "optim"
require "cutorch"
require "network"
require "seboost"

--batch_size = 256

input_patchsz = 16


number_of_images = 2000*256
number_of_validation_images = 500*256
batch_size = 256

optim_method = "adam"
training_log = optim.Logger('train_adam_batch_256.log')
validation_log = optim.Logger('validation_batch_256.log')

training_log:setNames{'Training Loss'}
validation_log:setNames{'Validation Loss'}


train_inputs = torch.load("train_inputs_tensor.t7")
train_outputs = torch.load("train_outputs_tensor.t7")

train_inputs = train_inputs:double():mul(2./255.):add(-1):cuda()


train_outputs = train_outputs:double():mul(2./255.):add(-1):cuda()


--print(train_inputs)
validation_inputs = torch.load("validation_inputs_tensor.t7")
validation_outputs = torch.load("validation_outputs_tensor.t7")


  validation_inputs = validation_inputs:double():mul(2./255.):add(-1):cuda()



  validation_outputs = validation_outputs:double():mul(2./255.):add(-1):cuda()


count = 0
mlp = create_network()
print(mlp)

convert_in = nn.Sequential():add(nn.Reshape(input_patchsz*input_patchsz))
--partable = nn.ParallelTable()
--partable:add(nn.Reshape(input_patchsz*input_patchsz))
--partable:add(nn.Reshape(input_patchsz*input_patchsz))
--convert_in:add(partable)
--convert_in:add(nn.JoinTable(1,1))
convert_in:cuda()

convert_out = nn.Reshape(output_patchsz*output_patchsz):cuda()


x,dl_dx = mlp:getParameters()
criterion = nn.MSECriterion()
--print(x[2])
--print(dl_dx)
x:cuda()
dl_dx:cuda()
criterion:cuda()

initConfig = { --Initial Configuration for all models
                 learningRate = 0.00001,
                 --weightDecay = opt.weightDecay,
                 --momentum = opt.momentum,
                 --learningRateDecay = opt.learningRateDecay
            }

optimConfig = {
               optMethod = optim.adam,
               optConfig = initConfig,
               sesopBatchSize = 20,
               maxEval = 20,
               histSize = 20,
               isCuda = true,
               momentum = 0.9,
               sesopUpdate = 500,
               anchorPoints = nil
            }

  useAnchor = true
  if useAnchor == true then
    optimConfig.anchorPoints = torch.Tensor{900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,25,20,15,10,5}
  end

  if optimConfig.anchorPoints then --If anchors exist
    optimConfig.anchorsSize = optimConfig.anchorPoints:size(1)
  else
    optimConfig.anchorsSize = 0
  end

state = {

}

state.aOpt = torch.zeros(optimConfig.histSize+optimConfig.anchorsSize)
state.dirs = torch.zeros(x:size(1), optimConfig.histSize+optimConfig.anchorsSize)

local function single_epoch(mlp,criterion,train_inputs,train_outputs,number_of_images,batch_size,x,dl_dx)

  local function feval(x_new)
    --if x~= x_new then
      x:copy(x_new)
    --end
    --print(x[1])
    dl_dx:zero()
    pred_outputs = mlp:forward(inputs)
    --print(pred_outputs:size())
    loss = criterion:forward(pred_outputs,outputs)
    --print(loss)
    grad_outs = criterion:backward(pred_outputs,outputs)
    grad_ins = mlp:backward(inputs, grad_outs)
    return loss, dl_dx
  end


  epoch = 1
  total_loss = 0
  for i = 1, number_of_images,batch_size do

    --print(i)

    if i + batch_size < number_of_images then
    inputs = convert_in:forward(train_inputs[{{i,i+batch_size},{},{}}])
    else
    inputs = convert_in:forward(train_inputs[{ {i,number_of_images},{},{} }])
    end
    --print(inputs:size())
    --print(image_1:size())
    --outputs = convert_in:forward(train_outputs[i])
     outputs = inputs
    --print(im_1[{ i,{},{} }])


    anchorPoints = false

    optimConfig.sesopData = inputs
    optimConfig.sesopLabels = outputs
    state.itr = epoch
    state.dirs = state.dirs

    --local _, errs = optim.seboost(feval, x, optimConfig, state)
    count = count + 1
    epoch = epoch + 1
    optim_params = {learningRate = 0.00001}
    local _,errs = optim.adam(feval,x,optim_params)
    total_loss = total_loss + errs[1]
    --print(errs[1])
  end
return total_loss/number_of_images

end

local function validation_epoch(mlp,criterion,train_inputs,train_outputs,number_of_images,batch_size,x,dl_dx)
  validation_loss = 0
  for i = 1, number_of_images,batch_size do

    if i + batch_size < number_of_images then
      inputs = convert_in:forward(train_inputs[{ {i,i+batch_size},{},{}}])
    else
       inputs = convert_in:forward(train_inputs[{ {i,number_of_images},{},{} }])
     end
    --outputs = convert_in:forward(train_outputs[i])
    outputs = inputs
    --print(im_1[{ i,{},{} }])

    --inputs = convert_in:forward({ image_1, image_3 })
    --outputs = convert_out:forward(image_2[{{}, {sx, ex}, {sy, ey}}])

    pred_outputs = mlp:forward(inputs)
    err = criterion:forward(pred_outputs,outputs)



   validation_loss = validation_loss + err
    --print(errs[1])
  end

  validation_loss = validation_loss/number_of_images

  return validation_loss
end


print("TRAINING_LOSS".."     ".."VALIDATION LOSS")



for iter = 1, 100 do
  mlp:training()
  training_loss = single_epoch(mlp,criterion,train_inputs,train_outputs,number_of_images,batch_size,x,dl_dx)
  training_log:add{torch.log(training_loss)}
  mlp:evaluate()
  validation_loss = validation_epoch(mlp,criterion,validation_inputs,validation_outputs,number_of_validation_images,batch_size,x,dl_dx)
  validation_log:add{torch.log(validation_loss)}
  print(training_loss.."    "..validation_loss)
  if iter%1000 == 0 then
    torch.save("mlp_seboost"..iter..".t7",mlp)
  end
end

training_log:plot()
validation_log:plot()


torch.save("mlp_plain_adam_seboost.t7",mlp)
