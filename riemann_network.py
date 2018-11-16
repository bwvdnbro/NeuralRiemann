import numpy as np
import pylab as pl

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

np.random.seed(42)

data = np.loadtxt("training_data.txt")
x = data[:,:6]
y = np.zeros((x.shape[0], 5))
y[data[:,6] == 0, 0] = 1.
y[data[:,6] == 1, 1] = 1.
y[data[:,6] == 2, 2] = 1.
y[data[:,6] == 3, 3] = 1.
y[data[:,6] == 4, 4] = 1.

ref_result = np.round(y[:,1] + 2. * y[:,2] + 3. * y[:,3] + 4. * y[:,4])

hidden_N = 8
hidden_w = np.random.normal(size = (x.shape[1], hidden_N))
hidden_bias = np.random.normal(size = hidden_N)

hidden_z = np.dot(x, hidden_w) + hidden_bias
hidden_a = sigmoid(hidden_z)

output_w = np.random.normal(size = (hidden_N, y.shape[1]))
output_bias = np.random.normal(size = y.shape[1])

output_z = np.dot(hidden_a, output_w) + output_bias
output_a = sigmoid(output_z)

eta = 0.0001
for t in range(10000):
#  batch = np.random.randint(0, x.shape[0], 100)
#  batch = np.unique(batch)
  batch = range(x.shape[0])

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

ofile = open("network_output.txt", "w")
ofile.write("# expected\toutput\n")
for ie, io in zip(ref_result, result):
  ofile.write("{0}\t{1}\n".format(ie, io))
