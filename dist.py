import matplotlib
matplotlib.use('Agg')
import numpy as np
import mnist_classifier
import glob
import matplotlib.pyplot as plt
import pdb

images = []

x = glob.glob('ganWithAutoencoder_model1/*')

for loc in x:
    y = np.load(loc)
    loc = 'processed_ganWithAutoencoder_model1/' + loc.split('/')[1].split('.')[0] + '.npy'
    np.save(loc, y)
    y = np.reshape(y, [784])
    images.append(y)

images = np.array(images)
stats = mnist_classifier.summary_statistics(images)
predictions = mnist_classifier.predict(images)
predictions = np.argmax(predictions, axis=1)
for i, prediction in enumerate(predictions):
    if prediction == 8:
        loc = x[i]
        print(loc)

print(stats)


def plotStats(stats):
    digits = [i for i in range(10)]
    fix, ax = plt.subplots()

    rects = ax.bar(digits, stats, color='b')
    
    ax.set_xlabel('Digit')
    ax.set_ylabel('Num Generated Examples')
    ax.set_title('Class Distribution of Generated Images')

    xticklabels = tuple([str(digit) for digit in digits])
    ax.set_xticks(digits)
    ax.set_xticklabels(xticklabels)
    #print(xticklabels)
    #ax.set_xticklabels(xticklabels)
    plt.savefig('statsWithClassFeaturesAuto.png')

plotStats(stats)
