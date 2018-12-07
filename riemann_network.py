import numpy as np
import pylab as pl
import argparse

hidden1_N = 6
hidden2_N = 10
eta = 0.0001
epoch = 10000
Nbatch = 500
decay = 0.1

argparser = argparse.ArgumentParser()
argparser.add_argument("--N1", "-m", action = "store", default = hidden1_N,
                       type = int)
argparser.add_argument("--N2", "-n", action = "store", default = hidden2_N,
                       type = int)
argparser.add_argument("--eta", "-e", action = "store", default = eta,
                       type = float)
argparser.add_argument("--epoch", "-t", action = "store", default = epoch,
                       type = int)
argparser.add_argument("--Nbatch", "-b", action = "store", default = Nbatch,
                       type = int)
argparser.add_argument("--decay", "-d", action = "store", default = decay,
                       type = float)
args = argparser.parse_args()

hidden1_N = args.N1
hidden2_N = args.N2
eta = args.eta
epoch = args.epoch
Nbatch = args.Nbatch
decay = args.decay

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

np.random.seed(42)

data = np.loadtxt("training_data.txt")

x = data[:,:6]
# change data: normalize all values
#data[:,1] /= data[:,0]
#data[:,3] -= data[:,2]
#data[:,5] /= data[:,4]

y = np.zeros((x.shape[0], 5))
y[data[:,6] == 0, 0] = 1.
y[data[:,6] == 1, 1] = 1.
y[data[:,6] == 2, 2] = 1.
y[data[:,6] == 3, 3] = 1.
y[data[:,6] == 4, 4] = 1.

ref_result = np.round(y[:,1] + 2. * y[:,2] + 3. * y[:,3] + 4. * y[:,4])

hidden1_w = np.random.normal(size = (x.shape[1], hidden1_N))
hidden1_bias = np.random.normal(size = hidden1_N)
hidden2_w = np.random.normal(size = (hidden1_N, hidden2_N))
hidden2_bias = np.random.normal(size = hidden2_N)
output_w = np.random.normal(size = (hidden2_N, y.shape[1]))
output_bias = np.random.normal(size = y.shape[1])

batch = np.random.randint(0, x.shape[0], Nbatch)
batch = np.unique(batch)
#batch = range(x.shape[0])

hidden1_z = np.dot(x[batch,:], hidden1_w) + hidden1_bias
hidden1_a = sigmoid(hidden1_z)

hidden2_z = np.dot(hidden1_a, hidden2_w) + hidden2_bias
hidden2_a = sigmoid(hidden2_z)

output_z = np.dot(hidden2_a, output_w) + output_bias
output_a = sigmoid(output_z)

for t in range(epoch):
  diff = output_a - y[batch,:]
  if t%10 == 0:
    print t, np.multiply(diff, diff).sum()
  Ca = output_a - y[batch,:]
  # quadratic cost function
#  output_delta = np.multiply(Ca, output_a * (1. - output_a))
  # cross-entropy
  output_delta = Ca
  hidden2_delta = np.multiply(np.dot(output_delta, output_w.T),
                              hidden2_a * (1. - hidden2_a))
  hidden1_delta = np.multiply(np.dot(hidden2_delta, hidden2_w.T),
                              hidden1_a * (1. - hidden1_a))

  output_w *= (1. - eta * decay)
  output_w -= eta * np.dot(hidden2_a.T, output_delta) / len(batch)
  output_bias -= eta * output_delta.sum(0) / len(batch)

  hidden2_w *= (1. - eta * decay)
  hidden2_w -= eta * np.dot(hidden1_a.T, hidden2_delta) / len(batch)
  hidden2_bias -= eta * hidden2_delta.sum(0) / len(batch)

  hidden1_w *= (1. - eta * decay)
  hidden1_w -= eta * np.dot(x[batch,:].T, hidden1_delta) / len(batch)
  hidden1_bias -= eta * hidden1_delta.sum(0) / len(batch)

  batch = np.random.randint(0, x.shape[0], Nbatch)
  batch = np.unique(batch)
#  batch = range(x.shape[0])

  hidden1_z = np.dot(x[batch,:], hidden1_w) + hidden1_bias
  hidden1_a = sigmoid(hidden1_z)

  hidden2_z = np.dot(hidden1_a, hidden2_w) + hidden2_bias
  hidden2_a = sigmoid(hidden2_z)

  output_z = np.dot(hidden2_a, output_w) + output_bias
  output_a = sigmoid(output_z)

# compute full set prediction
hidden1_z = np.dot(x, hidden1_w) + hidden1_bias
hidden1_a = sigmoid(hidden1_z)

hidden2_z = np.dot(hidden1_a, hidden2_w) + hidden2_bias
hidden2_a = sigmoid(hidden2_z)

output_z = np.dot(hidden2_a, output_w) + output_bias
output_a = sigmoid(output_z)

result = np.round(output_a[:,1] + 2. * output_a[:,2] + 3. * output_a[:,3] +
                  4. * output_a[:,4])

ofile = open("network_weights.txt", "w")
ofile.write("# layer 1:\n")
ofile.write("w:\n")
for wrow in hidden1_w:
  for wcol in wrow:
    ofile.write("\t{0}".format(wcol))
  ofile.write("\n")
ofile.write("b: {0}\n".format(hidden1_bias))
ofile.write("# layer 2:\n")
ofile.write("w:\n")
for wrow in hidden2_w:
  for wcol in wrow:
    ofile.write("\t{0}".format(wcol))
  ofile.write("\n")
ofile.write("b: {0}\n".format(hidden2_bias))
ofile.write("# layer 3:\n")
ofile.write("w:\n")
for wrow in output_w:
  for wcol in wrow:
    ofile.write("\t{0}".format(wcol))
  ofile.write("\n")
ofile.write("b: {0}\n".format(output_bias))
ofile.close()

ofile = open("network_output.txt", "w")
ofile.write("# expected\toutput\n")
for ie, io in zip(ref_result, result):
  ofile.write("{0}\t{1}\n".format(ie, io))
