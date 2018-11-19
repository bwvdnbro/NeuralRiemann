#! /bin/bash

for N1 in {5..12}
do
  for N2 in {5..15}
  do
    python riemann_network.py --N1 $N1 --N2 $N2 --epoch 100000 --Nbatch 100 \
      --eta 0.01
    python plot_matrix.py --N1 $N1 --N2 $N2
    mv network_output.png network_output_"$N1"_"$N2".png
    mv network_weights.txt network_weights_"$N1"_"$N2".txt
  done
done

