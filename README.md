# ganAdditionalFeatures

## Minibatch class distribution as additional feature for discriminator

The files involved in training these networks are ```gan-script``` (training a normal network) and ```gan_dist_class_features.py``` (using the class distribution as additional features). ```dist.py``` is used to evaluate the class distribution of the generated images and ```mnist_classifier.py``` is used to classify MNIST images.


## Testing of effects of additional features on mode collapse with Gaussian mixture distributions

The files involved in data generation are ```gen_data.py``` (generates the data) and ```gan-script-gaussians.py```, which trains the GAN on the data generated from ```gen_data```. ```gan_dist_class_features_gaussians.py``` trains the GAN on Gaussians from ```gen_data``` with additional class distribution data data.
