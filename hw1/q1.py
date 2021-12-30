import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


point_set = [np.random.rand(100, 2**i) for i in range(10)]
mean = []
std = []

for sample in point_set:
    dst = []
    for i in range(100):
        for j in range(i+1, 100):
            dst.append((distance.euclidean(sample[i], sample[j]))**2)
    mean.append(np.mean(dst))
    std.append(np.std(dst))
    
    
d = [i**2 for i in range(10)]

plt.plot(d, mean)
plt.plot(d, std)
plt.xlabel("dimension")
plt.ylabel("distance")
plt.legend(["mean", "std"])
plt.title("Mean and Std of Euclidean Distances as a Function of Dimension")
plt.show()

