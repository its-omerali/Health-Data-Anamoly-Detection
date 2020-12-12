import inline as inline
import matplotlib
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from scipy.interpolate import make_interp_spline,BSpline
from scipy.interpolate import interp1d
import seaborn as sns

total_data = pd.read_csv('ROC_Calculation.csv')


#Close all previous graphs
plt.close('all')


fig = plt.figure()
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.title('ROC Curves for DBSCAN algorithm with various Epsilon values')
positions = (0,0.2,0.4,0.6,0.8,1)
plt.xlabel('False Positives P(T+ | D-)')
plt.ylabel('Sensitivity P(T+ | D+)')

x1=total_data['seq1_x']
y1=total_data['seq1_y']
x1_smooth = total_data.groupby('seq1_x').mean()
y1_smooth = total_data.groupby('seq1_x').mean()
x2=total_data['seq2_x']
y2=total_data['seq2_y']
x3=total_data['seq3_x']
y3=total_data['seq3_y']
x4=total_data['seq4_x']
y4=total_data['seq4_y']
plt.style.use('grayscale')
plt.grid()
ax = fig.gca()
# This portion will try to add scientific grid to the plots
# Set axis ranges; by default this will put major ticks every 25.
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(0.2))
ax.yaxis.set_minor_locator(AutoMinorLocator(0.2))
# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle='--')
#End of Scientific Grid
plt.xticks(positions)
plt.plot(x4,y4, linestyle='-',color='black', linewidth='1.15', marker='.', label='Epsilon 4 | Area 77.38')
plt.fill_between(x1, y1, color='blue', alpha = 0.02)
plt.plot(x1,y1, linestyle='-', color='blue',linewidth='1.15', marker='.', label='Epsilon 5 | Area 89.21')

plt.plot(x2,y2, linestyle='-', color='olive',linewidth='1.15', marker='.', label='Epsilon 6 | Area 82.03')
plt.plot(x3,y3, linestyle='-', color='crimson',linewidth='1.15', marker='.', label='Epsilon 7 | Area 79.32')


plt.legend(fancybox=True)

plt.savefig('ROC_Curves.png', dpi=300)
plt.show()

