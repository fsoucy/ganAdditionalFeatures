import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = glob.glob('generated_images_withClassFeaturesMod1/*npy')
for img_loc in x:
    y = np.load(img_loc)
    loc = img_loc.split('/')[1].split('.')[0]
    plt.imsave('processed_generated_images_withClassFeaturesMod1/' + loc + '.jpg', y)

