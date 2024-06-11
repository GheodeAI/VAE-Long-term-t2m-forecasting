"""The code to run the AE+MLP methodology to forecast the 2-meter air temperatures in 7 different cities which are
    given as coordinates of longitude and latitude. All the climate data we are using were downloaded from Copernicus
    CCS, which can be accessed with following link.

    https://climate.copernicus.eu/

    1st part: the AE
    2nd part: the MLP (hence AE+MLP)
"""


import prepare_data
import reading
import time
import Algorithms
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv


def statistical_analysis(true, pred):
    """This is to calculate the mean squared error (mse) and mean absolute error (mae) of forecasted data.

        Args:
            true (float vector): The real temperature data in given time period for given city.
            pred (float vector): The forecasted temperature data in given time period for given city.
                The number of elements of 'pred' must be the same as 'true'.
        
        Returns:
            mse, mae (list): A list of 2 values, the first one calculated mse, the second one the calculated mae.
    """
    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    return mean_squared_error(true, pred).__round__(3), mean_absolute_error(true, pred).__round__(3)


# Create the list of cities and their properties
cities = [['paris',       2003, 49, 42],
          ['cordoba',     1995, 38, 36],
          ['athens',      1987, 38, 64],
          ['frankfurt',   2006, 50, 49],
          ['szczecin',    1994, 53, 55],
          ['sofia',       2007, 43, 63],
          ['smolensk',    2010, 55, 72]]

for city in cities:
    """Initial for loop to run over the cities
    
        First, the climate data is selected, preprocessed and stashed into numpy matrices. 
    """

    lat = city[2] # extract all the information from the list
    long = city[3]
    Devs = prepare_data.Deviations()
    year_of_pred = city[1]
    variables = ['z', 'sst', 't2m']  # swvl1
    preprocessing = ['normalize', 'normalize', 'normalize']

    (x_train1, x_test1) = prepare_data.prepare_data(variables, preprocessing, plt_hist=True, border=year_of_pred)
    (x_train1, __) = prepare_data.fit_delay(x_train1, delay=0)

    for iter in range(10):
        """Next, the model is built, compiled and trained.
            
            Each model is saved in the end in the form of h5.
        """

        print('Currently, the ' + str(iter) + '. iteration is being run for city of ' + str(city[0]) +
              ', lat: ' + str(lat) + ', long: ' + str(long) + ', year_of_pred: ' + str(year_of_pred))

        tf.keras.backend.clear_session()
        timer = round(time.time())

        AE = True
        model_1 = Algorithms.AE_PRED(80, 64, AE=AE, channels_in=len(variables), channels_out=len(variables))
        model_1.compile(loss='mse')
        model_1.fit(x_train1, x_train1, epochs=5000, batch_size=128, verbose=0)
        name_1 = city[0] + '_' + str(iter) + '_AE_' + str(AE) + '_' + str(timer) + '.h5'
        model_1.autoencoder.save(name_1)

        dl_anomalies_test = reading.propagate(name=name_1, data_in=x_test1[:, 0],
                                              data_out=x_test1[:, 0],
                                              variables=variables, anomalies=False, plot=True)

        stat_list = []

        for delay in range(1, 5):
            """Employ the MLP part.
                
                Again, each model is saved in the end in the form of h5.
            """

            # Prepare the DL deviations by forecasting the AE, the data is then input into the MLP
            (climate_anomalies_train, climate_anomalies_test) = Devs.prepare_deviations(['t2m'])
            (__, climate_anomalies_train_out) = prepare_data.fit_delay(climate_anomalies_train, delay=delay)
            prepare_data.plot_hist(climate_anomalies_train, climate_anomalies_test, name='climate_anomalies')

            (x_train, x_test) = prepare_data.prepare_data(variables, preprocessing, border=year_of_pred)
            (x_train, __) = prepare_data.fit_delay(x_train, delay=delay)  # implement the delay
            latent_space_train = model_1.encoder.predict(x_train)[2]

            # print(city[0] + '  ' + str(x_test.shape)) for testing purposes
            model_2 = Algorithms.MLP(64)
            model_2.compile(loss='mse')
            model_2.fit(latent_space_train, climate_anomalies_train_out[:, lat, long, :],
                        epochs=5000, batch_size=128, verbose=0)
            name_2 = city[0] + '_' + str(iter) + '_MLP_' + str(timer) + '_delay_' + str(delay) + '.h5'
            model_2.mlp.save(name_2)

            """Finally, the forecasts are done.
                
                Results are presented in graphical and tabular forms.
            """
            x_test = x_test[:, 0] # Select the first possible date which is equal to the year of desired forecasting
            latent_space_test = model_1.encoder.predict(x_test)[2]
            t2m_pred = model_2.mlp.predict(latent_space_test)
            t2m_mask = np.zeros((52, 80, 80, 1))
            t2m_mask[:, lat, long, 0] = t2m_pred[:, 0]
            t2m_pred = t2m_mask

            t2m_pred = Devs.inverse_deviations(t2m_pred[delay:], delay=delay)
            prepare_data.plot_hist(t2m_pred, name='t2m_anom_pred')

            # Result's graphs - tabular and graphical
            (climate, ground_truth) = prepare_data.prepare_data(['t2m'], preprocessing=['none'], border=year_of_pred)
            climate = climate.mean(axis=1, keepdims=True)
            ground_truth = ground_truth[:, 0, lat, long, 0]
            climate = climate[delay:, 0, lat, long, 0]

            persistence = np.zeros((52 - delay,))
            persistence[:] = ground_truth[:(52 - delay)]
            ground_truth = ground_truth[delay:]

            t2m_pred = t2m_pred[:, 0, lat, long, 0]

            """A series of 3 cross-checks to verify the integrity of the code."""
            # To cross-check the ground truth with raw data for given year and city
            # plt.figure()
            # plt.plot(ground_truth)
            # plt.grid()
            # plt.savefig('ground_truth.png')

            # To cross-check the inverted normalisation of the data
            # plt.figure()
            # (__, t2m_test) = prepare_data.prepare_data(['t2m'], ['none'], border=2003)
            # plt.plot(t2m_test[:, 0, 40, 40, :], 'k')
            # plt.plot(Devs.inverse_deviations(climate_anomalies_test)[:, 1, 40, 40, :], 'r--')
            # plt.savefig('inverse_deviations_check.png')

            # To cross-check the location
            # (loc_test, __) = prepare_data.prepare_data(['t2m'], preprocessing=['none'], border=year_of_pred)
            # loc_test = loc_test.mean(axis=1, keepdims=True)
            # plt.figure()
            # loc_test[0, 0, lat, long, 0] = 1
            # plt.imshow(loc_test[0, 0, :, :, 0], origin='lower', cmap='jet')
            # plt.savefig('location_check.png')


            """Outputs and exports of results"""
            # First one is the figure, sketched in matplotlib
            plt.figure()
            plt.plot(np.concatenate((np.ones(delay) * np.nan, ground_truth - persistence), axis=0), 'g--', label='persistence')
            plt.plot(np.concatenate((np.ones(delay) * np.nan, ground_truth - climate), axis=0), 'k--', label='climatology')
            plt.plot(np.concatenate((np.ones(delay) * np.nan, ground_truth - t2m_pred), axis=0), 'r', label='AE+MLP')
            plt.title('Lead time: ' + str(delay) + ', year: ' + str(year_of_pred))
            plt.ylabel('Error [\N{degree sign} Celsius]')
            plt.xlabel('Week')
            plt.grid()
            plt.legend()
            plt.savefig(city[0] + '_' + str(iter) + '_2nd_AE_MLP_' + str(delay) + '_' + str(timer) + '.png')

            # The second one is the statistical table, which comes in the .txt format
            stats = statistical_analysis(ground_truth, t2m_pred)
            stat_list.append(stats[0])
            stat_list.append(stats[1])

            with open(city[0]+'_stats_'+str(iter)+'AE_'+str(AE)+'_MLP_'+str(timer)+'_delay_'+str(delay)+'.txt', 'w') as f:
                f.write('Stats file for further research.\n')
                f.write('\n**********\n')
                f.write('\npersistence\n')
                f.write(str(statistical_analysis(ground_truth, persistence)))   # Write the mse and mae
                f.write('\nclimate\n')
                f.write(str(statistical_analysis(ground_truth, climate)))       # Write the mse and mae
                f.write('\nt2m\n')
                f.write(str(stats))
                f.write('\n**********\n')
                f.write('\nraw series gnd truth\n')
                f.write(str(ground_truth.round(4)))
                f.write('\nraw series persistence\n')
                f.write(str(persistence.round(4)))
                f.write('\nraw series climate\n')
                f.write(str(climate.round(4)))
                f.write('\nraw series t2m\n')
                f.write(str(t2m_pred.round(4)))

        ### Statistical file output
        with open('Stats_ae_mlp.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(stat_list)
        plt.close()
