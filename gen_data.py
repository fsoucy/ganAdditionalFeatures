import numpy as np
import matplotlib.pyplot as plt
#  k = number of gaussians
# n = total number of data points
def create_dataset_weighted(k,n) :
    means = np.linspace(0,4*k,k)
    weights = np.random.uniform(0,1,k)
    weights /= sum(weights)
    vars = np.tile([1],k)
    freqs = np.multiply(weights,n*1.0)
    freqs =[int(val) for val in freqs]
    points = np.array([np.random.normal(means[i],vars[i],freqs[i]) for i in range(k)]).flatten()
    data = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            data.append(points[i][j])
    np.save('weighted',data)
    print('done!')
    return data

# n = number of data points per class
def create_dataset_unweighted(k,n) :
    means = np.linspace(0,4*k,k)
    vars = np.tile([1],k)
    points = np.array([np.random.normal(means[i],vars[i],n) for i in range(k)]).flatten()
    np.save('unweighted',points)
    print('done!')
    return points

#create_dataset_unweighted(10,1000)
#create_dataset_weighted(10,10000)
data_weighted = np.load('weighted.npy')
data_unweighted = np.load('unweighted.npy')
plt.clf()
plt.hist(create_dataset_unweighted(10,10000),bins=500,histtype='step')
plt.show()
plt.clf()
plt.hist(create_dataset_weighted(10,10000),bins=500,histtype='step')
plt.show()
