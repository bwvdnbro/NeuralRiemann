import numpy as np
import pylab as pl

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

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

hidden_w = np.random.normal(size = (x.shape[1], 10))
hidden_bias = np.random.normal(size = 10)

hidden_z = np.dot(x, hidden_w) + hidden_bias
hidden_a = sigmoid(hidden_z)

output_w = np.random.normal(size = (10, y.shape[1]))
output_bias = np.random.normal(size = y.shape[1])

output_z = np.dot(hidden_a, output_w) + output_bias
output_a = sigmoid(output_z)

eta = 0.002
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
  hidden_delta = np.multiply(np.dot(output_delta, output_w.T),
                             hidden_a[batch,:] * (1. - hidden_a[batch,:]))

  output_w -= eta * np.dot(hidden_a[batch,:].T, output_delta)
  output_bias -= eta * output_delta.sum()

  hidden_w -= eta * np.dot(x[batch,:].T, hidden_delta)
  hidden_bias -= eta * hidden_delta.sum()

  hidden_z = np.dot(x, hidden_w) + hidden_bias
  hidden_a = sigmoid(hidden_z)

  output_z = np.dot(hidden_a, output_w) + output_bias
  output_a = sigmoid(output_z)

control = np.random.random((10000,6))
hidden_z = np.dot(control, hidden_w) + hidden_bias
hidden_a = sigmoid(hidden_z)
output_z = np.dot(hidden_a, output_w) + output_bias
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
