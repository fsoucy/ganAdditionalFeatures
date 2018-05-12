# ganAdditionalFeatures

## Minibatch class distribution as additional feature for discriminator

The files involved in training these networks are ```gan-script``` (training a normal network) and ```gan_dist_class_features.py``` (using the class distribution as additional features). ```dist.py``` is used to evaluate the class distribution of the generated images and ```mnist_classifier.py``` is used to classify MNIST images.