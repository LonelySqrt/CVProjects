import numpy as np

############################################################################
################### THIS HELPER IS ONLY FOR CONV WITH BN ###################
############################################################################
############### Note that: There are no parameters b anymore. ##############

def nn_f(X, W):
	# X : (m, n_prev)
	# W : (n, n_prev)
	Z = X @ W.T
	cache = (X, W)
	return Z, cache


def nn_b(dZ, cache):

	# dZ : (m, n)
	# cache : X, W
	X, W = cache
	m, _ = dZ.shape

	dW = dZ.T @ X / m
	dX = dZ @ W

	return dX, dW


def zero_pad(X, pad):

	#  X  : shape(m, n_H, n_W, n_C)
	# pad : scalar

	return np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0,0))


def conv_f(X, W, params):

	# X -> input,  shape : (m, in_H, in_W, in_C)
	# W -> filter, shape : (filters, f, f, in_C)
	# params: dict{'s':s, 'pad':pad}
	
	# Return:
	# Z -> conv output, shape : (m, out_H, out_W, filters)
	# cache -> Z, W (cache them for backward)

	filters, f, f, in_C = W.shape
	m, in_H, in_W, in_C = X.shape
	s, pad = params['s'], params['pad']

	out_H = int((in_H+2*pad-f)/s)+1
	out_W = int((in_W+2*pad-f)/s)+1

	X_pad = zero_pad(X, pad) if pad != 0 else X
	Z = np.zeros((m, out_H, out_W, filters), dtype=np.float32)

	for i in range(m):
		for h in range(out_H):
			for w in range(out_W):
				v_s, v_e, h_s, h_e = h*s, h*s+f, w*s, w*s+f
				x_slice = X_pad[i, v_s:v_e, h_s:h_e, :]
				Z[i,h,w,:] = ((W * x_slice).reshape(filters, -1).sum(1)).astype(np.float32)
			
	cache = (X, W, params)
	return Z, cache

def conv_b(dZ, cache):

	# dZ is the derivative
	# cache (X, W, params)

	X, W, params = cache
	m, n_H, n_W, _ = dZ.shape
	filters, f, f, _ = W.shape
	s, pad = params['s'], params['pad']
	
	dX = np.zeros(X.shape, dtype=np.float32)
	dW = np.zeros(W.shape, dtype=np.float32)

	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):
				v_s, v_e, h_s, h_e = h*s, h*s+f, w*s, w*s+f
				x_slice = X[i, v_s:v_e, h_s:h_e, :]
				dW += (x_slice * dZ[i,h,w,:][:, None, None, None]).astype(np.float32)
				dX[i, v_s:v_e, h_s:h_e, :] += np.sum(W * dZ[i,h,w,:][:, None, None, None], axis=0).astype(np.float32)
			
	return dX, dW/m


def pool_f(X, params, mode=0):
	# NOTE THAT: POOLING LAYER DOESN'T HAVE WEIGHTS
	# Input parameters:
	# X -> input shape:(m, in_H, in_W, in_C)
	# params -> dict{f, s, out_C}
	# mode -> max=0, avg=1, default:0

	# Return:
	# Z -> output shape:(m, out_H, out_W, out_C)
	# cache -> X, params, mode (sava them for backwards)

	m, in_H, in_W, in_C = X.shape
	f, s = params['f'], params['s']

	out_H = int((in_H-f)/s)+1
	out_W = int((in_W-f)/s)+1

	Z = np.zeros((m, out_H, out_W, in_C), dtype=np.float32)
	pool = np.max if mode == 0 else np.mean
	
	for i in range(m):
		x_single = X[i]
		for h in range(out_H):
			for w in range(out_W):
				v_s, v_e, h_s, h_e = h*s, h*s+f, w*s, w*s+f
				x_slice = x_single[v_s:v_e, h_s:h_e, :]
				Z[i,h,w,:] = pool(x_slice.reshape(-1, in_C), axis=0)

	cache = (X, params, mode)
	return Z, cache

def pool_b(dZ, cache):

	# mode: 0->max, 1->avg
	X, params, mode = cache
	m, n_H, n_W, n_C = dZ.shape
	f, s = params['f'], params['s']
	dX = np.zeros(X.shape)
	pool = max_pool_b if mode == 0 else avg_pool_b

	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):
				v_s, v_e, h_s, h_e = h*s, h*s+f, w*s, w*s+f
				dz_block = dZ[i, v_s, h_s, :]
				x_block = X[i, v_s:v_e, h_s:h_e, :]
				dX[i, v_s:v_e, h_s:h_e, :] = pool(dz_block, x_block)
	return dX


def max_pool_b(dz, x):
	# dz : (n_c,)
	# x  : (f,f,n_c)
	return (x == x.max(axis=(0,1))).astype(np.int16) * dz


def avg_pool_b(dz, x):
	# dz : (n_c,)
	# x  : (f,f,n_c)
	return np.full(x.shape, dz/(x.shape[0]**2))


def relu(Z):
	# Z : any shape
	return np.maximum(0, Z)


def relu_b(dA, Z):
	dZ = np.array(dA, copy=True)
	dZ[ Z<=0 ] = 0
	return dZ


def sigmoid(Z):
	# Z : any shape
	return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_b(dA, Z):
	return dA * sigmoid(Z) * (1-sigmoid(Z))

#########################################################
################ 2018.04.10 / 2018.04.20 ################
#########################################################

def bn_conv_f(X, gamma, beta, epsilon=1e-6):

	#  batch_norm
	#  between: (conv,pool) and (activation)
	#
	#     X : (m, n_H, n_W, n_C)
	#  beta : (n_C,)
	# gamma : (n_C,)
	#     Y : (m, n_H, n_W, n_C) <- output

	# NOTE THAT: ***********************************************************************************
	# We learn a pair of parameters gamma(k) and beta(k) per feature map, rather than per activation
	# **********************************************************************************************

	m, n_H, n_W, n_C = X.shape
	# dof : degree of freedom
	dof_mu = m*n_H*n_W
	dof_s = (m-1)*n_H*n_W

	mu = X.sum(axis=(0,1,2)) / dof_mu
	s = np.power(X-mu, 2).sum(axis=(0,1,2)) / dof_s
	X_norm = ((X-mu) / np.sqrt(s+epsilon)).astype(np.float32)
	Y = gamma*X_norm + beta

	cache = (mu, s, X, X_norm, gamma, beta, epsilon)
	return Y, cache


def bn_conv_b(dY, cache):

	# dY : (m, n_H, n_W, n_C)
	# cache : the forward cache
	m, n_H, n_W, n_C = dY.shape
	mu, s, X, X_norm, gamma, beta, epsilon = cache

	# dof : degree of freedom
	dof = m*n_H*n_W
	dof_s = (m-1)*n_H*n_W

	# calculate the derivation
	dX_norm = dY*gamma
	dbeta = dY.sum(axis=(0,1,2)) / dof
	dgamma = ((dY * X_norm).sum(axis=(0,1,2)) / dof).astype(np.float32)
	ds = -(dX_norm*(X-mu)*np.power(s+epsilon, -1.5)).astype(np.float32).sum(axis=(0,1,2)) / (2*dof_s)
	dmu = -(dX_norm*np.power(s+epsilon, -0.5).astype(np.float32) + 2*ds*(X-mu) / dof_s).sum(axis=(0,1,2)) / dof
	dX = dX_norm*np.power(s+epsilon, -0.5).astype(np.float32) + 2*ds*(X-mu) / dof_s + dmu / dof

	return dX, dgamma, dbeta

def bn_nn_f(X, gamma, beta, epsilon=1e-6):
	# X : (m, n)
	m, n = X.shape
	mu = X.sum(axis=0) / m
	s = np.power(X-mu, 2).sum(axis=0) / (m-1)
	X_norm = (X-mu) / np.sqrt(s+epsilon)
	Z = X_norm*gamma + beta

	cache = (mu, s, X, X_norm, gamma, beta, epsilon)
	return Z, cache

def bn_nn_b(dY, cache):
	
	#    dY : (m, n)
	# cache : (mu, s, X, X_norm, gamma, beta, epsilon)
	mu, s, X, X_norm, gamma, beta, epsilon = cache
	m, n = dY.shape

	# calculate the derivation
	dX_norm = dY*gamma
	dbeta = dY.sum(axis=(0)) / m
	dgamma = (dY*X_norm).sum(axis=(0)) / m
	ds = -(dX_norm*(X-mu)*np.power(s+epsilon, -1.5)).astype(np.float32).sum(axis=(0)) / (2*(m-1))
	dmu = -(dX_norm*np.power(s+epsilon, -0.5).astype(np.float32) + 2*ds*(X-mu)/(m-1)).sum(axis=(0)) / m
	dX = dX_norm*np.power(s+epsilon, -0.5).astype(np.float32) + 2*ds*(X-mu)/(m-1) + dmu/m

	return dX, dgamma, dbeta


def adam(params, grads, hps, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
	# combained momentun and RMSprob together

	# params : W,  gamma,  beta
	#    hps : Vdw, Vdgamma, Vdbeta, Sdw, Sdgamma, Sdbeta.
	W, gamma, beta = params
	keys = ['VdW', 'Vdgamma', 'Vdbeta', 'SdW', 'Sdgamma', 'Sdbeta']
	VdW, Vdgamma, Vdbeta, SdW, Sdgamma, Sdbeta = [hps[key] for key in keys]

	for l in W.keys():
		corr = 1-np.power(beta1, t)
		# update W
		VdW[l] = beta1*VdW[l] + (1-beta1)*grads[l]['dW']
		SdW[l] = beta2*SdW[l] + (1-beta2)*(grads[l]['dW']**2).astype(np.float32)
		grads[l]['dW'] -= lr*(VdW[l] / corr) / (np.sqrt(SdW[l] / corr)+epsilon)

		# update gamma
		Vdgamma[l] = beta1*Vdgamma[l] + (1-beta1)*grads[l]['dgamma']
		Sdgamma[l] = beta2*Sdgamma[l] + (1-beta2)*(grads[l]['dgamma']**2).astype(np.float32)
		grads[l]['dgamma'] -= lr*(Vdgamma[l] / corr) / (np.sqrt(Sdgamma[l] / corr)+epsilon)

		# update beta
		Vdbeta[l] = beta1*Vdbeta[l] + (1-beta1)*grads[l]['dbeta']
		Sdbeta[l] = beta2*Sdbeta[l] + (1-beta2)*(grads[l]['dbeta']**2).astype(np.float32)
		grads[l]['dbeta'] -= lr*(Vdbeta[l] / corr) / (np.sqrt(Sdbeta[l] / corr)+epsilon)