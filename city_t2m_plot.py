import matplotlib.pyplot as plt
import xarray as xr

font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

ds_X = xr.open_dataset('../North_1_deg_weekly_reduced.nc')

fig, (ax1, ax2) = plt.subplots(2, 1)
cordoba = ds_X.t2m.sel(latitude=38, longitude=-4, time=ds_X.time.dt.year == 1995, method='nearest')
paris = ds_X.t2m.sel(latitude=49, longitude=2, time=ds_X.time.dt.year == 2003, method='nearest')

ax1.plot(cordoba - 273)
ax2.plot(paris - 273)

ax1.grid()
ax2.grid()

ax1.axvline(x=28, color='r', linestyle='--')
ax1.axvline(x=30, color='r', linestyle='--')

ax2.axvline(x=31, color='r', linestyle='--')
ax2.axvline(x=33, color='r', linestyle='--')

ax1.title.set_text('Cordoba 1995')
ax2.title.set_text('Paris 2003')

ax1.set_ylabel(r'Temperature ($^\circ$C)')
ax2.set_ylabel(r'Temperature ($^\circ$C)')

ax1.set_xlabel(r'Weeks in the year')
ax2.set_xlabel(r'Weeks in the year')

plt.show()
