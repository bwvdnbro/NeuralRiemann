import numpy as np
import pylab as pl
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--N1", "-m", action = "store", default = -1,
                       type = int)
argparser.add_argument("--N2", "-n", action = "store", default = -1,
                       type = int)
args = argparser.parse_args()

output = np.loadtxt("network_output.txt")

size = np.zeros((5, 5))
for ie in range(5):
  for io in range(5):
    size[ie, io] = ((output[:,0] == ie) & (output[:,1] == io)).sum()

grid = np.meshgrid(*np.ogrid[0:5,0:5])
pl.scatter(grid[0], grid[1], s = size.T)

for ie in range(5):
  for io in range(5):
    pl.text(ie, io, "{0:.0f}".format(size[ie, io]),
            verticalalignment = "center",
            horizontalalignment = "center")

if args.N1 > 0:
  pl.text(0, 4.5, "N1: {0}".format(args.N1))
if args.N2 > 0:
  pl.text(1, 4.5, "N2: {0}".format(args.N2))
pl.xlim(-1, 5)
pl.ylim(-1, 5)
pl.xlabel("expected category")
pl.ylabel("predicted category")
pl.title("Neural network Riemann solver performance")
pl.tight_layout()
pl.savefig("network_output.png", dpi = 300)
