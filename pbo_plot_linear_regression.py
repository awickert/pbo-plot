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

plt.ion()
fig = plt.figure(figsize=(8,12))
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(pbo_data.Date, pbo_data['East (mm)'], 'o', color='gray')
plt.ylabel('East Displacement [mm]', fontsize=16)
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(pbo_data.Date, pbo_data['North (mm)'], 'o', color='gray')
plt.ylabel('North Displacement [mm]', fontsize=16)
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(pbo_data.Date, pbo_data['Vertical (mm)'], 'o', color='gray')
plt.ylabel('Vertical Displacement [mm]', fontsize=16)
plt.xlabel('Year', fontsize=16)

plt.tight_layout()

#############
## ANALYZE ##
#############

# Least-squares regression in Pandas
from pandas.stats.api import ols

# Create a new column of days since the start for regression
pbo_data['date_delta'] = (pbo_data['Date'] - pbo_data['Date'].min())  / np.timedelta64(1,'D')

# Residuals
res_x = ols(x=pbo_data['date_delta'], y=pbo_data['East (mm)'])
res_y = ols(x=pbo_data['date_delta'], y=pbo_data['North (mm)'])
res_z = ols(x=pbo_data['date_delta'], y=pbo_data['Vertical (mm)'])





