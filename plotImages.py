import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import figure, imshow, axis, imsave
from matplotlib.image import imread
import glob

list_of_files = glob.glob('generated_images/*png')[:4]

def showImagesHorizontally(list_of_files):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')
    imsave('thing.png', fig)

showImagesHorizontally(list_of_files)
