import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dLossFake = np.load('classFeaturesData/dLossFakesWithoutClassFeatures.npy')
dLossReal = np.load('classFeaturesData/dLossRealsWithoutClassFeatures.npy')

dLossFakeW = np.load('classFeaturesData/dLossFakesWithClassFeatures.npy')
dLossRealW = np.load('classFeaturesData/dLossRealsWithClassFeatures.npy')

dLossFakeM = np.load('classFeaturesData/dLossFakesWithClassFeaturesMod1.npy')
dLossRealM = np.load('classFeaturesData/dLossRealsWithClassFeaturesMod1.npy')

gLoss = np.load('classFeaturesData/gLossWithoutClassFeatures.npy')
gLossW = np.load('classFeaturesData/gLossWithClassFeatures.npy')
gLossM = np.load('classFeaturesData/gLossWithClassFeaturesMod1.npy')

plt.plot(dLossFake, label='dLossFake without class features')
plt.plot(dLossReal, label='dLossReal without class features')
#plt.plot(dLossFakeW, label='dLossFake with class features')
#plt.plot(dLossRealW, label='dLossReal with class features')
plt.plot(dLossFakeM, label='dLossFake with class features and modified training')
plt.plot(dLossRealM, label='dLossReal with class features and modified training')
#plt.plot(gLoss, label='Generator loss without class features')
#plt.plot(gLossW, label='Generator loss with class features')
#plt.plot(gLossM, label='Generator loss with class features and modified training')

plt.xlabel('Iterations')
plt.ylabel('Discriminator Loss')

plt.title('Discriminator Loss in Training')

plt.legend()

plt.savefig('dLossVsMod.png')
