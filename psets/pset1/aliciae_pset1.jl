#=
Alicia M. Elliott
18.337J Pset 1
February 2026
aliciae@mit.edu

Note to whomever is grading these: 
This is the first time I'm using Julia, so I'm picking up the syntax and conventions as I go! 
Before grad school I worked professionally in scientific computing, mostly using C++/CUDA/FORTRAN, 
so if I make some unexpected choices in the ways I implement my code, that might explain why. 
I've been out of practice for a couple years now, so I'm trying to get back into the good habits I learned while working.

I apologize in advance for the excessively verbose comments. 
I've learned the hard way to err on the side of verbosity when explaining my code to future-me :)
=#
using LinearAlgebra


#=
Problem 1
=#
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


#= 
Problem 2
=#
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

	return [Z[1], dZ[1]]	
end

xy = rand(Float64,2)
dxy = [0.0001, 0.0001]
	
print(NN_derivative(xy[1], xy[2], dxy[1], dxy[2]))


#=
Problem 3
=#
#function gradient_descent(f, df, u0, alpha, verbose=true)
function gradient_descent(df, u0, alpha, verbose=true)
	#f_init = f(u0)
	u_n = u0
	for i in 1:1000
		# get gradient at initial point
		grad = df(u_n)
		# choose our epsilon to consider the algorithm converged, to make life simple I'm just choosing machine epsilon for float.
		if abs(maximum(grad)) > eps(Float64)
			u_n = u_n - grad .* alpha
		else 
			print("\n\nConverged after $(i) iterations with learning rate $(alpha), eps $(eps(Float64))!\n\t Function minimized at u_n = $(u_n)\n")
			
			return u_n
		end

		if verbose 
			print("\nIteration $(i)\t u_n $(u_n)    grad(u_n) $(grad)   ") 
		end

	end

end

function parabola(u)
	x = u[1]
	y = u[2]
	return x.^2 + y.^2
end

function grad_parabola(u)
	x = u[1]
	y = u[2]
	return [2. .* x, 2. .* y]
end

u0_in = [1.0, 1.0]
u_min = gradient_descent(grad_parabola, u0_in, 0.4, false)

# true minimum is [0.0, 0.0] at which point the value f(0,0) = 0.0. We'll compare within machine precision for float. 
print(isapprox(parabola(u_min), 0.0; atol=eps(Float64), rtol=0))


#=
Problem 4

For the representation of the Poisson equation with Dirichlet BCs on the boundaries x=0, x=1, y=0, y=1
I like the representation presented in the IEEE paper reference that is linked in the pset,  https://ieeexplore.ieee.org/document/712178
The representation in the paper includes generalized Dirichlet BCs without defining them as zero. 
Trial function u_t(x,y) = A(x,y) + x*(1-x) * y *(1-y) * NN(x,y,w,b) (eq. 22 from the paper)
	A(x,y) is chosen for the boundary conditions, see eq. 23 in paper. 
	In this case, we've defined that the function value is zero on all domain boundaries, 
	so A(x,y) becomes zero when you fill in the values into eq. 23:
		u(0,y)=f_0(y) = 0
		u(1,y) = f_1(y) = 0
		u(x,0) = g_0(x) = 0
		u(x,1) = g_1(x) = 0
	So, the function approximator I'll use is defined now by:
		u_t(x,y) = x * (1-x) * y (1-y) * NN(x,y,w,b)
	The x*(1-x)*y*(1-y) terms take care of the Dirichlet BC enforcement (at least, I think they do? I've convinced myself with some math, but this feels suspiciously easy...)

	Then, to define the loss function which will determine when the weights satisfy the PINN


	So... I admittedly started this pset later than I should've. Here's what I *think* I'm supposed to do, though am unsure. 
	I think the loss function is supposed to be implemented by minimizing Sum_i [ d^2u(x_i,y_i)/dx^2 + d^2u(x_i,y_i)/dy^2 - (-sin(pi*x_i)*sin(pi*y_i)) ]^2
		where the second partials we substitute in the trial function with the NN and NN_deriv (x and y components, so Jacobian). 
		i.e. du/dx = (1-2x)* y*(y-1) * NN(x_i,y_i) + x(x-1)y(y-1)*NN_deriv_x(x_i,y_i)
		so then d^2u/dx^2 = d/dx (du/dx)
			= NN_deriv_x(x,y)(1-2x)y(1-y) + (-2x)y(1-y)*NN(x,y) 
				+ (1-2x)*y(y-1)*NN_deriv_x(x,y) + x(1-x)y(1-y)NN_deriv_deriv_x(x,y_)
		Using this expression, I'd need to write another function for the second derivatives of the neural network implementation,
		as was done with NN_derivative. So, we're re-implementing the NN and NN_deriv with the second derivatives too.
		I'm also restructuring the input a bit, so the function can take in weights and biases (which I should've done for the first ones too, but oh well.)

		The partials expressions for the second derivs can be simplified to:
			d2u/dx^2 = y*(1-y) [2*NN_deriv_x(x,y)(1-2x) - 2x*NN(x,y) + x(1-x)*NN_second_deriv_x(x,y)]
			d^2u/dy^2 = x*(1-x) [2*NN_deriv_y(x,y)(1-2y) - 2y*NN(x,y) + y(1-y)*NN_second_deriv_y(x,y)]

=#

function NN_second_derivative(x, y, theta)
	X = [x, y]
	W = theta[1]
	b = theta[2]

	# Again, as in function NN. We're taking in 2 inputs, using 2 hidden layers of size 16 each, and 1 output, hence the dims of the weights and bias arrays. 
	#W = [randn(16,2),randn(16,16),randn(1,16)]
	#b = [zeros(16),zeros(16),zeros(1)]

	# Same as in NN function, but splitting up into composite functions to apply the partial chain rule to each composite in the next step. 
	A1 = W[1]*X + b[1]
	A2 = W[2]*tanh.(A1) + b[2]
	A3 = W[3]*tanh.(A2) + b[3]
	Z = A3

	# derivative of tanh(z) is 1 - tanh^2(z), or sech^2(z)
	dA1_dX = W[1]
	dA2_dA1 = W[2] * sech.(A1).^2
	dA3_dA2 = W[3] * sech.(A2).^2

	# Compute the first derivatives
	dZ_dX = reshape(dA3_dA2,1,1) * reshape(dA2_dA1,1,16) * dA1_dX

	# To prep for the second derivatives, "split" first derivative expression into composite functions
	# This is unnecessary, but for the sake of nomenclature and tracking my math, I want to keep the variables clear. 
	B1 = dA1_dX # = W[1]
	B2 = dA2_dA1 # = W[2] * sech.(B1).^2
	B3 = dA3_dA2 # = W[3] * sech.(B2).^2

	# derivative of sech^2(z) = -2 sech^2(z)tanh(z)
	# so this is the second derivatives of our NN function. 
	dB1_dX = 1.
	dB2_dB1 = -2. * W[2] * sech.(B1).^2 .*tanh.(B1)
	dB3_dB2 = -2. * W[3] * sech.(B2).^2 .* tanh.(B2)

	# Compute the second derivatives now!
	d2Z_dX2 =  reshape(dB3_dB2,1,16) * reshape(dB2_dB1,16,2) * dB1_dX

	return [Z[1], dZ_dX, d2Z_dX2]	
end

function my_poisson_loss_function(x, y, theta)
	# Note: I don't think this code will work as intended because of the hard-coded matrix sizes I have in the NN_derivs functions.
	# Ideally I'd rewrite it so it can just vectorize everything. Alas, I did not start early enough, so we're going with this. 
	NN_z = NN_second_derivative(x, y, theta)
	Z_NN = NN_z[1]
	NN_derivs = NN_z[2] 
	NN_second_derivs = NN_z[3]

	# Note: I'm sure there's a smarter matrix way to do this calculation, 
	# but I do not trust my brain at this second to figure it out and get it right. 
	# This is less optimal, but it gets the job done (hopefully). 
	d2u_dx2 = y.*(1 - y).* (2. .* NN_derivs[1] .* (1. - 2. .* x) - 2. .* x * Z_NN + x.*(1. - x) * NN_second_derivs[1])
	d2u_dy2 = x.*(1 - x).* (2. .* NN_derivs[2] .* (1. - 2. .* y) - 2. .* y * Z_NN + y.*(1. - y) * NN_second_derivs[2])

	fxy = sin.(pi .* x) .* sin.(pi .* y)

	losses = (d2u_dx2 + d2u_dy2 - fxy).^2
	loss = sum(losses)
	return loss
end


xy = rand(Float64,2)
dxy = [0.0001, 0.0001]

Wts = [randn(16,2),randn(16,16),randn(1,16)]
biases = [zeros(16),zeros(16),zeros(1)]


# Okay, so now let's implement! Discretize the domain in x and y to, let's say 20 points in each, from 0 to 1. 
xvals = range(0.0, 1.0, length=20)
yvals = range(0.0, 1.0, length=20)

# Now we're going to implement using gradient descent and the neural network we defined!
# Initial random weights and zero biases
theta0 = [Wts, biases]

# our function "gradient" here is the loss function we've defined. 
# theta_n+1 = theta_n - grad(L(theta_n))*alpha

# so the gradient descent will update the NN parameters, compute the new weights/biases using the loss function and learning rate,
# until the loss function is minimized. For each iteration, we call the NN_second_derivative to get the new derivatives for the current params.

#= 
As I don't have time to finish the code properly before the deadline, I'll just explain the rest. 

So.. the way I wrote the gradient descent function inputs, I'd have to rework it a bit to have this work outright for the function calls to work in this.
But the concept is here. I'm going to submit this as-is so that it's on time, and keep working on it on my own. 
To implement it with this loss function, the gradient descent function needs to take in the x and y arrays as well as theta0, 
so that the loss function can be computed for the whole domain by inputting the x, y, and the updated weights. 
e.g. 
	theta_PINN = gradient_descent(my_poisson_loss_function, x, y, theta0, 0.1)
 
So the way it will work, each iteration the gradient descent will update the weights and biases (theta) by 
	theta_n+1 = theta_n - grad(L(theta_n))*alpha, 
where L(theta_n) is the loss function my_poisson_loss_function(x, y, theta_n)
	Once the parameters for the weights and biases in theta are sufficient for the loss function to sum to less than epsilon, 
	we can consider our neural network to be converged as a function estimator that solves the Poisson equation as we defined it. 
I don't know how many iterations it would take, but the learning rate should be moderate, i.e. if it's too large it will not find the minimum, 
as the weights/biases will be changed by a large amount and likely will struggle to find the minimized values. 
If the learning rate is too small, it will likely take many, many iterations to converge on the minimum. 
Likewise, I'm not sure how many x and y points are needed to sample to get this appropriately converged without being computationally gross. 
But too few points and it will struggle to "learn" the approximation of the physics, too many and it will be computationally expensive. 

I'll start the next pset earlier :)

=#