module Pset1

export NN, NN_derivative

using LinearAlgebra

# This function for the "3-layer" multilayer perceptron contains an input layer which takes in 2 values, 2 hidden layers with size 16 each, and 1 output layer returning a singular float64 value. 
function NN(x::Float64, y::Float64)::Float64
	# Take in 2 floats as inputs, return 1 float output.
	# Vectorize the inputs to make the math more linear-algebra-y. 
	# I could take in a vector instead to skip this part, but the scientific programmer in me prefers static typing when I can. Ideally I'd use the StaticArray library and enforce a size-2 float64 vector input and size-1 float64 vector output, but following the constraints of the pset explicitly forbidding use of other libraries, we will do it this way instead!
	X = [x, y]

	# We're taking in 2 inputs, using 2 hidden layers of size 16 each, and 1 output, hence the dims of the weights and bias arrays. 
	W = [randn(16,2),randn(16,16),randn(1,16)]
	b = [zeros(16),zeros(16),zeros(1)]
		
	# z = W3 sigma2 (W2 sigma1 (W1*X + b1) + b2) + b3, where sigma1/sigma2 are the tanh activation functions
	z::Float64 = first(W[3]*tanh.(W[2]*tanh.(W[1]*X+b[1]) + b[2]) + b[3])
		
	# For the previous line, I'm using "first" to grab the result from the 1-element Vector{Float64} to enforce my static typing requirement to return a singular float64 value. I'm sure there are cleaner ways to do this, but for readability's sake, this is what I'm going with. 

	return z
end

# Ensure the NN function can be called
vals = rand(Float64,2)
result = NN(vals[1],vals[2])
typeof(NN(vals[1],vals[2])) == Float64
	

function NN_derivative(x, y, dx, dy)
	X = [x, y]
	dX = [dx, dy]

	# Again, as in function NN. We're taking in 2 inputs, using 2 hidden layers of size 16 each, and 1 output, hence the dims of the weights and bias arrays. 
	W = [randn(16,2),randn(16,16),randn(1,16)]
	b = [zeros(16),zeros(16),zeros(1)]

	# Same as in NN function, but splitting up into composite functions to apply the partial chain rule to each composite in the next step. 
	A1 = W[1]*X + b[1]
	A2 = W[2]*tanh.(A1) + b[2]
	A3 = W[3]*tanh.(A2) + b[3]
	Z = A3

	# derivative of tanh(z) is 1 - tanh^2(z), or sech^2(z)
	dA1_dX = W[1]
	dA2_dA1 = W[2] * sech.(A1).^2
	dA3_dA2 = W[3] * sech.(A2).^2
	# dA2_dA1 = W[2]*(1. .- (tanh.(A1)).^2)
	# dA3_dA2 = W[3]*(1. .- (tanh.(A2)).^2)

	# reshape to be able to do math ~properly~	
	#tmp3 = reshape(dA3_dA2,1,1) * reshape(dA2_dA1,1,16) * reshape(dA1_dX * dX,16,1)

	dZ_dX = reshape(dA3_dA2,1,1) * reshape(dA2_dA1,1,16) * dA1_dX
	dZ = dZ_dX * dX 

	return [Z, dZ]	
end

xy = rand(Float64,2)
dxy = [0.0001, 0.0001]
	
NN_derivative(xy[1], xy[2], dxy[1], dxy[2])


end