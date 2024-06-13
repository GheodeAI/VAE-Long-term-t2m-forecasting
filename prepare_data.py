import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import statsmodels.api as sm
from statsmodels.tsa.tsatools import add_trend
import sys


def statistical_analysis(true, pred):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    return mean_squared_error(true, pred), mean_absolute_error(true, pred)


def plot_hist(*argv, **kwargs):
    plt.figure()
    length = len(argv)
    side = 1
    for var in argv:
        dim = var.shape[-1]
        for idx in range(dim):
            plt.subplot(dim, length, (idx * length) + side)
            plt.hist(var[..., -1].ravel())
            plt.ylabel('dim ' + str(idx))
        side += 1
    plt.savefig('histograms_' + kwargs['name'] + '.png')


def prepare_data(variables, preprocessing, border=2002, plt_hist=False, months=range(52)):
    x_train = np.zeros((52, border-1950, 80, 80, len(variables)))
    x_test = np.zeros((52, 2022-border, 80, 80, len(variables)))
    method = []
    for var in range(len(variables)):
        method.append(Data(variable=variables[var], scale=preprocessing[var],
                           border=border, plt_hist=plt_hist, months=months))
        (train, test) = method[var].return_data()
        x_train[:, :, :, :, [var]], x_test[:, :, :, :, [var]] = train, test
    return x_train, x_test


class Data:
    def __init__(self, variable, scale, border=2002, plt_hist=False, months=range(52)):
        self.variable = variable
        self.months = months
        self.plt_hist = plt_hist
        self.border = border
        self.scale = scale
        self.scale_1_param = []
        self.scale_2_param = []
        self.x_train = []
        self.x_test = []
        self._get_data()

    def _get_data(self):
        ds_X = xr.open_dataset('../North_1_deg_weekly_reduced.nc')
        size = ds_X.to_array().shape
        self.x_train = np.zeros((len(self.months), self.border-1950, size[-2], size[-1], 1))
        self.x_test = np.zeros((len(self.months), 2022-self.border, size[-2], size[-1], 1))
        for year in range(1950, self.border):
            data_train = np.array(ds_X[self.variable].sel(time=ds_X.time.dt.year == year))
            self.x_train[:, year - 1950, :, :, 0] = data_train[self.months]
        for year in range(self.border, 2022):
            data_test = np.array(ds_X[self.variable].sel(time=ds_X.time.dt.year == year))
            self.x_test[:, year - self.border, :, :, 0] = data_test[self.months]
        if self.scale:  # if the scale is not False, do the scaling
            self._scale()

        self.x_train[np.isnan(self.x_train)] = .5
        self.x_test[np.isnan(self.x_test)] = .5

    def return_data(self):
        return self.x_train, self.x_test

    def _scale(self):
        if self.scale == 'normalize':
            self.scaler_1_param = np.nanmax(self.x_train, axis=(0, 1, 2, 3))  # MAX
            self.scaler_2_param = np.nanmin(self.x_train, axis=(0, 1, 2, 3))  # MIN

            self.x_train = (self.x_train - self.scaler_2_param) / (self.scaler_1_param - self.scaler_2_param)
            self.x_test = (self.x_test - self.scaler_2_param) / (self.scaler_1_param - self.scaler_2_param)

        elif self.scale == 'standardize':
            self.scaler_1_param = np.nanmean(self.x_train, axis=(0, 1, 2, 3))  # MEAN
            self.scaler_2_param = np.nanstd(self.x_train, axis=(0, 1, 2, 3))   # STD

            self.x_train = (self.x_train - self.scaler_1_param) / (6 * self.scaler_2_param) + .5
            self.x_test = (self.x_test - self.scaler_1_param) / (6 * self.scaler_2_param) + .5

        elif self.scale == 'none':
            pass

    def unscale(self, data):
        if self.scale == 'normalize':
            data = data * (self.scaler_1_param - self.scaler_2_param) + self.scaler_2_param
        elif self.scale == 'standardize':
            data = (data - .5) * (6 * self.scaler_2_param) + self.scaler_1_param
        return data



    # def scale_data(self):
    #     elif prep == 'image':
    #         x_test[:,:,:,:,idx] = (x_test[:,:,:,:,idx] - np.nanmean(np.nanmean(x_test[:,:,:,:,idx], axis=2, keepdims=True), axis=3, keepdims=True)) / 6 / \
    #                  np.nanstd(np.nanstd(x_test[:,:,:,:,idx], axis=2, keepdims=True), axis=3, keepdims=True) + .5
    #         x_train[:,:,:,:,idx] = (x_train[:,:,:,:,idx] - np.nanmean(np.nanmean(x_train[:,:,:,:,idx], axis=2, keepdims=True), axis=3, keepdims=True)) / 6 /\
    #                   np.nanstd(np.nanstd(x_train[:,:,:,:,idx], axis=2, keepdims=True), axis=3, keepdims=True) + .5
    #     elif prep == 'regress':
    #         # linear regression, y = a * x + b
    #         # (x_train, x_train_out) = fit_delay(x_train, delay=0)
    #         for lat in range(80):
    #             for long in range(80):
    #                 for week in range(months[-1]-months[0]):  # years
    #                     # Calculate linear regression for each pixel, through weeks
    #                     y = x_train[week, :, lat, long, idx]
    #                     mdl = sm.OLS(y, add_trend(range(52), trend='c')).fit()  # detrend the series
    #                     r = y - mdl.predict(add_trend(range(52), trend='c'))  # deviations that arise from detrended val
    #                     r_std = r.std()
    #                     # digitize and normalize the classes
    #                     r = np.digitize(r, bins=[-r_std * 3, -r_std * 2, -r_std, r_std, r_std * 2, r_std * 3]) / 5.0
    #                     x_train[week, :, lat, long, idx] = r
    #
    #                     y = x_test[week, :, lat, long, idx]
    #                     r2 = y - mdl.predict(add_trend(range(52, 72), trend='c'))  # deviations that arise from detrended val
    #                     # digitize and normalize the classes
    #                     r2 = np.digitize(r2, bins=[-r_std * 3, -r_std * 2, -r_std, r_std, r_std * 2, r_std * 3]) / 5.0
    #                     x_test[week, :, lat, long, idx] = r2
    #
    #                     # plt.plot(mdl.predict(add_trend(range(72), trend='c')))
    #                     # plt.plot(np.concatenate([x_train[week, :, lat, long, idx], x_test[week, :, lat, long, idx]]))
    #                     # plt.savefig('test.png')
    #     elif prep == 'normalize':
    #         maxi = np.nanmax(x_train[:, :, :, :, idx], axis=(0, 1, 2, 3))
    #         mini = np.nanmin(x_train[:, :, :, :, idx], axis=(0, 1, 2, 3))
    #         mean = np.nanmean(x_train[:, :, :, :, idx], axis=(0, 1, 2, 3))
    #         std = np.nanstd(x_train[:, :, :, :, idx], axis=(0, 1, 2, 3))
    #         # x_test[:, :, :, :, idx] = (x_test[:, :, :, :, idx] - mini) / (maxi - mini)
    #         # x_train[:, :, :, :, idx] = (x_train[:, :, :, :, idx] - mini) / (maxi - mini)
    #         x_test[:, :, :, :, idx] = (x_test[:, :, :, :, idx] - mean) / (6 * std) + .5
    #         x_train[:, :, :, :, idx] = (x_train[:, :, :, :, idx] - mean) / (6 * std) + .5
    #
    #     elif prep == 'none':  # no transformation
    #         pass
    #     idx += 1
    #
    # x_train[np.isnan(x_train)] = .5
    # x_test[np.isnan(x_test)] = .5
    #
    # if plt_hist:
    #     plot_hist(x_train, x_test, name='input')
    #
    # return x_train, x_test


class Deviations:
    def __init__(self):
        self.mean = []
        self.std = []
        self.mini = []
        self.maxi = []
        self.diff_mean = []

    def prepare_deviations(self, variables, months=range(52), border=2002):
        ds_X = xr.open_dataset('../North_1_deg_weekly_reduced.nc')
        size = ds_X.to_array().shape
        x_train = np.zeros((len(months), border-1950, size[-2], size[-1], len(variables)))
        x_test = np.zeros((len(months), 1, size[-2], size[-1], len(variables)))
        for year in range(1950, border):
            data_train = np.array(ds_X[variables].sel(time=ds_X.time.dt.year == year).to_array())
            x_train[:, year - 1950, :, :, :] = np.moveaxis(data_train, 0, -1)[months]
        for year in range(border, border+1):
            data_test = np.array(ds_X[variables].sel(time=ds_X.time.dt.year == year).to_array())
            x_test[:, year - 2002, :, :, :] = np.moveaxis(data_test, 0, -1)[months]

        x_train[np.isnan(x_train)] = 0
        x_test[np.isnan(x_test)] = 0

        self.diff_mean = np.nanmean(x_train[:, :20], axis=1, keepdims=True)

        x_test = x_test - self.diff_mean
        x_train = x_train - self.diff_mean

        self.mini = x_train.min(axis=(0, 1), keepdims=True)
        self.maxi = x_train.max(axis=(0, 1), keepdims=True)
        self.mean = x_train.mean(axis=(0, 1), keepdims=True)
        self.std = x_train.std(axis=(0, 1), keepdims=True)

        if False:  # norm
            x_train = (x_train - self.mini) / (self.maxi - self.mini)
            x_test = (x_test - self.mini) / (self.maxi - self.mini)
        else:  # std
            x_train = (x_train - self.mean) / (24 * self.std) + .5
            x_test = (x_test - self.mean) / (24 * self.std) + .5

        # x_train[np.isnan(x_train)] = 0.5
        # x_test[np.isnan(x_test)] = 0.5

        return x_train, x_test

    def inverse_deviations(self, array, delay):
        return (array[:, np.newaxis] - .5) * (24 * self.std) + self.mean + self.diff_mean[delay:]


def fit_delay(climate, delay, sliding_window=False, window_len=16):
    weeks = climate.shape[0]
    years = climate.shape[1]
    no_of_lat = climate.shape[2]
    no_of_long = climate.shape[3]
    input = np.zeros(((weeks-delay)*years, no_of_lat, no_of_long, climate.shape[-1]))
    output = np.zeros(((weeks-delay)*years, no_of_lat, no_of_long, climate.shape[-1]))

    n = weeks - delay  # define the range of the observations, how many of them to exclude
    for year in range(years):
        input[year * n:(year + 1) * n] = climate[:n, year]
        output[year * n:(year + 1) * n] = climate[delay:, year]

    if sliding_window:
        input = np.mean(input, axis=(1, 2))
        output = np.mean(output, axis=(1, 2))
        input_sliced = np.zeros((input.shape[0] - window_len, window_len, input.shape[-1]))
        output_sliced = np.zeros((output.shape[0] - window_len, window_len, output.shape[-1]))
        for i in range(input.shape[0]-window_len):
            input_sliced[i] = input[i:(i + window_len)]
            output_sliced[i] = output[i:(i + window_len)]
        input = input_sliced
        output = output_sliced
    return input, output


def standardize(data):
    mean = data.mean(axis=(tuple(range(len(data.shape) - 1))))  # calculate mean for each dim
    std = data.std(axis=(tuple(range(len(data.shape) - 1))))  # calculate std for each dim

    return (data - mean) / (6 * std) + .5


def climate_prediction(data):
    return data.mean(axis=1, keepdims=True)
