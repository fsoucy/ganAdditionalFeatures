# ganAdditionalFeatures

## Minibatch class distribution as additional feature for discriminator

The files involved in training these networks are ```gan-script``` (training a normal network) and ```gan_dist_class_features.py``` (using the class distribution as additional features). ```dist.py``` is used to evaluate the class distribution of the generated images and ```mnist_classifier.py``` is used to classify MNIST images.


## Testing of effects of additional features on mode collapse with Gaussian mixture distributions

The files involved in data generation are ```gen_data.py``` (generates the data) and ```gan-script-gaussians.py```, which trains the GAN on the data generated from ```gen_data```. ```gan_dist_class_features_gaussians.py``` trains the GAN on Gaussians from ```gen_data``` with additional class distribution  data.

We break down the various folders as follows: 

```1d_single_mode``` and ```1d_multi_mode``` contain the code for data generation and GANs trained on 1d single Gaussians and 1d mixtures of Gaussians respectively.  Analogously, ```2d_single_mode``` and ```2d_multi_mode``` contain the code for data generation and GANs trained on 2d single Gaussians and 2d mixtures of Gaussians respectively.

```2d_multi_mode_additional_features``` contains general code corresponding to the augmentation of the ```2d_multi_mode``` code to allow for the input of additional features.  ```2d_multi_mode_dist_known```  contains the files corresponding to the experiment in which the features augmenting the data are calculated directly from prior knowledge of the generating distribution.  ```2d_multi_mode_dist_unkown``` contains code corresponding to the second case, in which the features augmenting the data are calculated directly from the distribution, without any additional information about the distribution responsible for generating the data.

