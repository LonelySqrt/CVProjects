import numpy as np

def nn_f(X, W, b):
	# X : (m, n_prev)
	# W : (n, n_prev)
	# b : (n, 1)
	Z = X @ W.T + b
	cache = {'X':X, 'W':W}
	return Z, cache


def nn_b(dZ, cache):

	# dZ : (m, n)
	# cache : X, W

	X, W = cache['X'], cache['W']
	m, _ = dZ.shape

	try:
		db = dZ.sum(0) / m
		dW = dZ.T @ X / m
		dX = dZ @ W
	except Exception as e:
		print(dZ.shape)
		print(W.shape)
		assert False
	
	return dX, dW, db


def zero_pad(X, pad):

	#  X  : shape(m, n_H, n_W, n_C)
	# pad : scalar
	return np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=(0,0))


def conv_f(X, W, b, params):

	# X -> input,  shape : (m, in_H, in_W, in_C)
	# W -> filter, shape : (filters, f, f, in_C)
	# b -> bias,   shape : (filters, 1)
	# params: dict{'s':s, 'pad':pad}

	# Z -> conv output, shape : (m, out_H, out_W, filters)
	# cache -> Z, W, b (cache them for backward)

	filters, f, f, in_C = W.shape
	m, in_H, in_W, in_C = X.shape
	s, pad = params['s'], params['pad']

	out_H = int((in_H+2*pad-f)/s)+1
	out_W = int((in_W+2*pad-f)/s)+1

	X_pad = zero_pad(X, pad) if pad != 0 else X
	Z = np.zeros((m, out_H, out_W, filters), dtype = np.float32)

	for i in range(m):
		for h in range(out_H):
			for w in range(out_W):
				v_s, v_e, h_s, h_e = h*s, h*s+f, w*s, w*s+f
				x_slice = X_pad[i, v_s:v_e, h_s:h_e, :]
				Z[i,h,w,:] = (W * x_slice).reshape(filters, -1).sum(1)
			
	cache = {'X':X, 'W':W, 'params':params}

	return Z, cache

def conv_b(dZ, cache):

	# dZ is the derivative
	# cache (X, W, params)

	X, W, params = cache['X'], cache['W'], cache['params']
	m, n_H, n_W, _ = dZ.shape
	filters, f, f, _ = W.shape
	s, pad = params['s'], params['pad']
	
	dX = np.zeros(X.shape, dtype=np.float32)
	dW = np.zeros(W.shape, dtype=np.float32)
	db = np.zeros((filters,), dtype=np.float32)

	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):

				v_s, v_e, h_s, h_e = h*s, h*s+f, w*s, w*s+f
				x_slice = X[i, v_s:v_e, h_s:h_e, :]  #(f,f,c)

				db += dZ[i,h,w,:]
				dW += x_slice * dZ[i,h,w,:][:, None, None, None]
				dX[i, v_s:v_e, h_s:h_e, :] += np.sum(W * dZ[i,h,w,:][:, None, None, None], axis=0)
			
	return dX, dW/m, db/m


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
				x_slice = x_single[v_s:v_e, h_s:h_e, :]    # x_slice: (f, f, in_C)
				Z[i,h,w,:] = pool(x_slice.reshape(-1, in_C), axis=0)
	
	cache = {'X':X, 'params':params, 'mode':mode}
	return Z, cache

def pool_b(dZ, cache):
	# mode: 0->max, 1->avg
	X, params, mode = cache['X'], cache['params'], cache['mode']
	m, n_H, n_W, n_C = dZ.shape
	f, s = params['f'], params['s']
	dX = np.zeros(X.shape, dtype=np.float32)
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
	return np.full(x.shape, dz/(x.shape[0]**2), dtype=np.float32)


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