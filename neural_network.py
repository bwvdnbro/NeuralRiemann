import numpy as np
import pylab as pl

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

eta = 0.001
layer1 = 5
layer2 = 10

np.random.seed(42)

x = np.random.random((1000,6))

ia = x[:,0] < 0.2
ib = (x[:,0] >= 0.2) & (x[:,0] < 0.4)
ic = (x[:,0] >= 0.4) & (x[:,0] < 0.6)
id = (x[:,0] >= 0.6) & (x[:,0] < 0.8)
ie = x[:,0] >= 0.8

y = np.zeros((x.shape[0], 5))
y[ia,0] = 1.
y[ib,1] = 1.
y[ic,2] = 1.
y[id,3] = 1.
y[ie,4] = 1.

ref_result = np.round(y[:,1] + 2. * y[:,2] + 3. * y[:,3] + 4. * y[:,4])

hidden1_w = np.random.normal(size = (x.shape[1], layer1))
hidden1_bias = np.random.normal(size = layer1)

hidden1_z = np.dot(x, hidden1_w) + hidden1_bias
hidden1_a = sigmoid(hidden1_z)

hidden2_w = np.random.normal(size = (layer1, layer2))
hidden2_bias = np.random.normal(size = layer2)

hidden2_z = np.dot(hidden1_a, hidden2_w) + hidden2_bias
hidden2_a = sigmoid(hidden2_z)

output_w = np.random.normal(size = (layer2, y.shape[1]))
output_bias = np.random.normal(size = y.shape[1])

output_z = np.dot(hidden2_a, output_w) + output_bias
output_a = sigmoid(output_z)

for t in range(100000):
  batch = np.random.randint(0, x.shape[0], 100)

  diff = output_a - y
  if t%10 == 0:
    print t, np.multiply(diff, diff).sum()
  Ca = output_a[batch,:] - y[batch,:]
  # quadratic cost function
#  output_delta = np.multiply(Ca, output_a * (1. - output_a))
  # cross-entropy
  output_delta = Ca
  hidden2_delta = np.multiply(np.dot(output_delta, output_w.T),
                              hidden2_a[batch,:] * (1. - hidden2_a[batch,:]))
  hidden1_delta = np.multiply(np.dot(hidden2_delta, hidden2_w.T),
                              hidden1_a[batch,:] * (1. - hidden1_a[batch,:]))

  output_w -= eta * np.dot(hidden2_a[batch,:].T, output_delta)
  output_bias -= eta * output_delta.sum()

  hidden2_w -= eta * np.dot(hidden1_a[batch,:].T, hidden2_delta)
  hidden2_bias -= eta * hidden2_delta.sum()

  hidden1_w -= eta * np.dot(x[batch,:].T, hidden1_delta)
  hidden1_bias -= eta * hidden1_delta.sum()

  hidden1_z = np.dot(x, hidden1_w) + hidden1_bias
  hidden1_a = sigmoid(hidden1_z)

  hidden2_z = np.dot(hidden1_a, hidden2_w) + hidden2_bias
  hidden2_a = sigmoid(hidden2_z)

  output_z = np.dot(hidden2_a, output_w) + output_bias
  output_a = sigmoid(output_z)

control = np.random.random((10000,6))
hidden1_z = np.dot(control, hidden1_w) + hidden1_bias
hidden1_a = sigmoid(hidden1_z)
hidden2_z = np.dot(hidden1_a, hidden2_w) + hidden2_bias
hidden2_a = sigmoid(hidden2_z)
output_z = np.dot(hidden2_a, output_w) + output_bias
output_a = sigmoid(output_z)

result = np.round(output_a[:,1] + 2. * output_a[:,2] + 3. * output_a[:,3] +
                  4. * output_a[:,4])

pl.plot(control[:,0], result, ".")
pl.plot([0., 0.2], [0., 0.])
pl.plot([0.2, 0.4], [1., 1.])
pl.plot([0.4, 0.6], [2., 2.])
pl.plot([0.6, 0.8], [3., 3.])
pl.plot([0.8, 1.], [4., 4.])
pl.gca().axvline(x = 0.2)
pl.gca().axvline(x = 0.4)
pl.gca().axvline(x = 0.6)
pl.gca().axvline(x = 0.8)
pl.show()
