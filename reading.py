import tensorflow as tf
from tensorflow import keras
from keras import layers
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def gen_images(name, data_in, data_out, predicted, variables):
    plt.figure()
    for i in range(round(np.floor(data_out.shape[0] / 10))):
        idx = i*10
        no = 5 + data_out.shape[-1]
        for j in range(idx, idx+10):  # draw first n-samples, where n=[0,1,..10]
            for k in range(data_out.shape[-1]):
                plt.subplot(10, no, (j-idx)*no+k+1)
                plt.imshow(data_out[j, :, :, k], vmin=0, vmax=1, cmap='jet', origin='lower')
            for k in range(data_out.shape[-1]):
                plt.subplot(10, no, (j-idx)*no+k+6)
                plt.imshow(data_out[j, :, :, k] - predicted[j, :, :, k], vmin=-.1, vmax=.1, cmap='jet', origin='lower')
            # plt.subplot(10, no, (j-idx)*no+no-2)
            # plt.imshow(data_out[j, :, :, 0], vmin=0, vmax=1, cmap='jet', origin='lower')
            # plt.subplot(10, no, (j-idx)*no+no-1)
            # plt.imshow(predicted[j, :, :, 0], vmin=0, vmax=1, cmap='jet', origin='lower')
            # plt.subplot(10, no, (j-idx)*no+no)
            # plt.imshow(data_out[j, :, :, 0] - predicted[j, :, :, 0], vmin=-.1, vmax=.1, cmap='jet', origin='lower')

        for k in range(data_out.shape[-1]):
            plt.subplot(10, no, k+1)
            plt.title(variables[k])
        for k in range(data_out.shape[-1]):
            plt.subplot(10, no, k + 6)
            plt.title('dev')
        # plt.subplot(10, no, no-2)
        # plt.title('out')
        # plt.subplot(10, no, no-1)
        # plt.title('pred')
        # plt.subplot(10, no, no)
        # plt.title('dev')
        plt.savefig(str(i) + '_' + name + '_pack.png')


def propagate(name, data_in, data_out=None, variables=None, anomalies=False, plot=False):
    model = keras.models.load_model(name)
    if anomalies:
        predicted = data_in - model.predict(data_in, batch_size=128) + .5
    else:
        predicted = model.predict(data_in)
    if plot:
        gen_images(name, data_in, data_out, predicted, variables)
    return predicted


def plt_climate(climate_anomalies):
    for i in range(52):
        plt.subplot(10, 10, i+1)
        plt.imshow(climate_anomalies[i, :, :, 0], vmin=0, vmax=1, cmap='jet', origin='lower')
    plt.savefig('climate_anomalies.png')

def plt_images(true, pred):
    plt.plot(true, 'k--')
    plt.plot(pred, 'r--')
    plt.savefig('img.png')
