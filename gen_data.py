import numpy as np

#  k = number of gaussians
# n = total number of data points
def create_dataset_weighted(k,n) :
    means = np.arange(0,20,k)
    weights = np.random.uniform(0,1,k)
    weights *= sum(weights)
    vars = np.tile([1],k)
    points = [np.random.normal(means[i],vars[i],weights[i]*n) for i in range(k)]
    np.save('weighted',points)
    print('done!')

# n = number of data points per class
def create_dataset_unweighted(k,n) :
    means = np.arange(0,20,k)
    vars = np.tile([1],k)
    points = [np.random.normal(means[i],vars[i],n) for i in range(k)]
    np.save('unweighted',points)
    print('done!')
