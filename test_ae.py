# This is only to test the pred_ae.py

import Algorithms
# from keras.utils.vis_utils import plot_model


AE = True
variables = ['z', 'sst', 't2m']

model_1 = Algorithms.AE_PRED(80, 64, AE=AE, channels_in=len(variables), channels_out=len(variables))
model_1.compile(loss='mse')
# plot_model(model_1, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print('Success')
