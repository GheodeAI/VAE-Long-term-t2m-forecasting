import matplotlib.pyplot as plt
import geopandas

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.boundary.plot(color='k', linewidth=1)
# plt.savefig('world_coastline.png')

plt.plot(2, 49, 'r.', markersize=15) # Paris
plt.plot(-4, 38, 'r.', markersize=15) # Cordoba
plt.plot(24, 38, 'r.', markersize=15) # Athens
plt.plot(9, 50, 'r.', markersize=15) # Frankfurt
plt.plot(15, 53, 'r.', markersize=15) # Sczeczin
plt.plot(23, 43, 'r.', markersize=15) # Sofia
plt.plot(32, 55, 'r.', markersize=15) # Smolensk

# plt.title('Selected locations', fontsize=16)
plt.xlim([-30, 60]) # longitude
plt.ylim([20, 70]) # latitude
plt.grid()
# plt.xlabel('Longitude', fontsize=16)
# plt.ylabel('Latitude', fontsize=16)


plt.show()