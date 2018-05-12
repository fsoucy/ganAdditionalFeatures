import numpy as np
import matplotlib.pyplot as plt
#  k = number of gaussians
# n = total number of data points
def create_dataset_weighted(k,n) :
    means = np.linspace(0,4*k,k)
    print(means)
    weights = np.random.uniform(0,1,k)
    weights *= sum(weights)
    print(weights)
    vars = np.tile([1],k)
    print(vars)
    freqs = np.multiply(weights,n*1.0)
    freqs =[int(val) for val in freqs]
    print(freqs)
    points = [np.random.normal(means[i],vars[i],freqs[i]) for i in range(k)]
    print(points)
    np.save('weighted',points)
    print('done!')
    return points

# n = number of data points per class
def create_dataset_unweighted(k,n) :
    means = np.linspace(0,4*k,k)
    print(means)
    vars = np.tile([1],k)
    print(vars)
    points = [np.random.normal(means[i],vars[i],n) for i in range(k)]
    np.save('unweighted',points)
    print('done!')
    return points

# plt.clf()
# plt.hist(create_dataset_unweighted(10,10000),bins=500,histtype='step')
# plt.show()

plt.clf()
plt.hist(create_dataset_weighted(10,10000),bins=500,histtype='step')
plt.show()
