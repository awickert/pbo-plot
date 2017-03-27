#! /usr/bin/env python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#################
## IMPORT DATA ##
#################

# Parse dates in first column of data file
pbo_data = pd.read_csv('P316.pbo.igs08.csv', header=11, parse_dates=[0], infer_datetime_format=True)
# Becasuse of whitespace in headers...
pbo_data = pbo_data.rename(columns=lambda x: x.strip())

##########
## PLOT ##
##########

#plt.ion() # Turn on for interactive plotting
fig = plt.figure(figsize=(8,12))
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(pbo_data.Date, pbo_data['East (mm)'], 'o', color='white')
plt.ylabel('East Displacement [mm]', fontsize=16)
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(pbo_data.Date, pbo_data['North (mm)'], 'o', color='white')
plt.ylabel('North Displacement [mm]', fontsize=16)
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(pbo_data.Date, pbo_data['Vertical (mm)'], 'o', color='white')
plt.ylabel('Vertical Displacement [mm]', fontsize=16)
plt.xlabel('Year', fontsize=16)

plt.tight_layout()

#############
## ANALYZE ##
#############

import statsmodels.api as sm

# Create a new column of days since the start for regression
pbo_data['date_delta'] = (pbo_data['Date'] - pbo_data['Date'].min()) \
                         / np.timedelta64(1,'D')
X = pbo_data['date_delta']
X = sm.add_constant(X)

# Residuals -- we know that we started at relative point (0,0,0), but
# have added in an offset just in case the GPS was bumped or slightly offset
# from its initial zeroing
fit_x = sm.OLS(pbo_data['East (mm)'], X).fit()
fit_y = sm.OLS(pbo_data['North (mm)'], X).fit()
fit_z = sm.OLS(pbo_data['Vertical (mm)'], X).fit()

ax1.plot(pbo_data['Date'], fit_x.params.values[1] * pbo_data['date_delta'] + \
         fit_x.params.values[0], 'k-', linewidth=2)
ax2.plot(pbo_data['Date'], fit_y.params.values[1] * pbo_data['date_delta'] + fit_y.params.values[0], 'k-', linewidth=2)
ax3.plot(pbo_data['Date'], fit_z.params.values[1] * pbo_data['date_delta'] + fit_z.params.values[0], 'k-', linewidth=2)

# At this point, time to analyze the structure of the residuals
res_x = pbo_data['East (mm)'] - \
       (fit_x.params.values[1] * pbo_data['date_delta'] \
        + fit_x.params.values[0])
res_y = pbo_data['North (mm)'] - \
       (fit_y.params.values[1] * pbo_data['date_delta'] \
        + fit_y.params.values[0])
res_z = pbo_data['Vertical (mm)'] - \
       (fit_z.params.values[1] * pbo_data['date_delta'] \
        + fit_z.params.values[0])

####################
## PLOT RESIDUALS ##
####################

# This is nearly a direct copy of the figure plotting above.
fig2 = plt.figure(figsize=(8,12))
ax21 = fig2.add_subplot(3, 1, 1)
ax21.plot(pbo_data.Date, res_x, 'o', color='white', zorder=1)
plt.hlines(0, pbo_data.Date.values[0], pbo_data.Date.values[-1], 'k', \
           linewidth=2, zorder=100)
plt.ylabel('East Displacement\nresidual [mm]', fontsize=16)
ax22 = fig2.add_subplot(3, 1, 2)
ax22.plot(pbo_data.Date, res_y, 'o', color='white', zorder=1)
plt.hlines(0, pbo_data.Date.values[0], pbo_data.Date.values[-1], 'k', \
           linewidth=2, zorder=100)
plt.ylabel('North Displacement\nresidual [mm]', fontsize=16)
ax23 = fig2.add_subplot(3, 1, 3)
ax23.plot(pbo_data.Date, res_z, 'o', color='white', zorder=1)
plt.hlines(0, pbo_data.Date.values[0], pbo_data.Date.values[-1], 'k', \
           linewidth=2, zorder=100)
plt.ylabel('Vertical Displacement\nresidual [mm]', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.tight_layout()

plt.show()

# Now what do you see in the residuals?

