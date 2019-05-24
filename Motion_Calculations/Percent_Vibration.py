'''
This file will take the CSVs from CDPM_for_loop and determine the residual
vibration and plot the differnce in percent vibration from the shaped and
unshaped responses. The end postion of the end-effector must be at 10,7.5 for
this program to work. You must also change n to make it the amount of csvs you
have.
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

filepath = '/Users/forrest/Documents/CRAWLAB-Student-Code/Forrest Montgomery/CDPM/Motion_Calculations/CSV/'

n=20
amps = np.array([(0,0)])
percent = np.array([0])
for i in range(n):
    resp = np.genfromtxt(filepath+'response'+str(i)+'.csv', delimiter=',')
    shaped = np.genfromtxt(filepath+'response_s'+str(i)+'.csv', delimiter=',')

    if shaped[:,0][1] >= 10:
        itemindex = np.where(shaped[:,0] >= 10)
    else:
        itemindex = np.where(shaped[:,0] <= 10)

    if resp[:,0][1] >= 10:
        itemindex_resp = np.where(resp[:,0] >= 10)
    else:
        itemindex_resp = np.where(resp[:,0] <= 10)

    shaped_ends = itemindex[:1][0][0]

    unshaped_ends = itemindex_resp[:1][0][0]

    beta_max = np.abs(np.max(resp[:,2][unshaped_ends:]))
    beta_min = np.abs(np.min(resp[:,2][unshaped_ends:]))
    beta_amp = np.abs(beta_max - beta_min)

    beta_shaped_max = np.abs(np.max(shaped[:,2][shaped_ends:]))
    beta_shaped_min = np.abs(np.min(shaped[:,2][shaped_ends:]))
    beta_shaped_amp = np.abs(beta_shaped_max - beta_shaped_min)

    percent_vib = (beta_shaped_amp / beta_amp) * 100
    amps = np.append(amps, [(beta_amp, beta_shaped_amp)], axis=0)
    percent = np.append(percent, [percent_vib])
    percent = np.nan_to_num(percent)
amps = amps[1:,:]

plt.figure(0)
plt.plot(percent)
plt.title('Percent Vibration')
plt.xlabel('Move Number')
plt.ylabel('Percent Vibration')
plt.show()

plt.figure(1)
plt.plot(amps[:,0], label='Unshaped')
plt.plot(amps[:,1], label='Shaped')
plt.title('Total Amplitude after Command')
plt.legend()
plt.xlabel('Move Number')
plt.ylabel('Amplitude')
plt.show()
