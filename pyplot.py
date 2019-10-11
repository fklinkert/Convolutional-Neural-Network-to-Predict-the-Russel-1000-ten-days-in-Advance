import csv
import numpy as np
import matplotlib.pyplot as plt

with open("write.csv", 'r') as f:
	fd = csv.reader(f,delimiter=",")
	fd = list(fd)

fd = fd[0]
fd = np.array(fd[:298], dtype = float)
yhat = fd[:149]
y = fd[149:298]
t = np.arange(0, 149, 1)
plt.plot(t, y, 'r', t, yhat, 'b')
plt.axis([0, 142, 100, 250])
plt.show()