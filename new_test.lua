require "image"
require "cunn"
require "nn"

local function rip2pieces(image1,mlp)

	local height = image1:size(1)
	local width  = image1:size(2)

	local recon_image = torch.zeros(height, width)

	for i = 1, height-16,1 do
		print(i)
		for j = 1, width-16,1 do
			patch_1  = image1[{ {i, i+15}, {j,j+15} }]

			test_patch_1 = patch_1:clone() -- Don't forget to clone()


			output = mlp:forward(patch_1:reshape(1,256):cuda())
			output = output:reshape(16,16):double()
			test_output = output:clone()

			image_out = test_output:add(1):mul(255./2.):byte()
			temp =	torch.add(recon_image[{ {i,i+15}, {j,j+15} }],output)
			recon_image[{ {i,i+15},{j,j+15} }]:copy(temp)

   		patch_image_1 = test_patch_1:add(1):mul(255./2.):byte()
		end
	end


	print(recon_image:size())
	out_image = recon_image:div(256):add(1):mul(255./2.):byte()
	return out_image

end

	local mlp = torch.load("mlp_seboost5000.t7")

		image1 = image.load("97.jpg",1,'byte')
		print(image1:size())

		--image1 = image1:reshape(image1:size(2),image1:size(3))
		print(image1:size())
		--image1 = image.scale(image1,image2:size(2)/2,image2:size(1)/2)
		--image2 = image.scale(image2,image2:size(2)/2,image2:size(1)/2)
		local image1 = image1:double():mul(2./255.):add(-1)

		out_image = rip2pieces(image1,mlp)

		image.save("noisy.png",image1:add(1):mul(255./2.):byte())
		image.save("denoised.png",out_image)
