#!/usr/bin/python
'''
title                  :Heatmaps.py
description            :This script takes the Natural frequency and Damping
                       ratios and plots the heatmaps. Also put in the area you
                       would like to estimate and it will compute the average
                       and the min and max.
author                 :Forrest
date                   :20160318
version                :1.0
usage                  :python Heatmaps.py
notes                  :
python_version         :3.5.1 |Anaconda 2.4.1 (x86_64)
#==============================================================================
'''
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

XY_L1L2_natflmh_damplmh = np.genfromtxt('XY_L1L2_natflmh_damplmh2.csv',
                                        delimiter=',')

'''
This code takes upper and lower bounds and pulls them out out the
larger CSV so you can take the averages for a better estimate.
'''
# for the circle
# upper_limit_x = 14
# lower_limit_x = 6

# upper_limit_y = 11.5
# lower_limit_y = 3.5

upper_limit_x = 19
lower_limit_x = 0

upper_limit_y = 14
lower_limit_y = 0

X = XY_L1L2_natflmh_damplmh[:,0]
Y = XY_L1L2_natflmh_damplmh[:,1]
L1 = XY_L1L2_natflmh_damplmh[:,2]
L2 = XY_L1L2_natflmh_damplmh[:,3]
nl = XY_L1L2_natflmh_damplmh[:,4]
nm = XY_L1L2_natflmh_damplmh[:,5]
nh = XY_L1L2_natflmh_damplmh[:,6]
dl = XY_L1L2_natflmh_damplmh[:,7]
dm = XY_L1L2_natflmh_damplmh[:,8]
dh = XY_L1L2_natflmh_damplmh[:,9]

itemindex = np.where(X >= lower_limit_x)
X_below = X[itemindex]
Y_below = Y[itemindex]
L1_below = L1[itemindex]
L2_below = L2[itemindex]
nl_below = nl[itemindex]
nm_below = nm[itemindex]
nh_below = nh[itemindex]
dl_below = dl[itemindex]
dm_below = dm[itemindex]
dh_below = dh[itemindex]

itemindex2 = np.where(X_below <= upper_limit_x)
X_range = X_below[itemindex2]
Y_range = Y_below[itemindex2]
L1_range = L1[itemindex2]
L2_range = L2[itemindex2]
nl_range = nl[itemindex2]
nm_range = nm[itemindex2]
nh_range = nh[itemindex2]
dl_range = dl[itemindex2]
dm_range = dm[itemindex2]
dh_range = dh[itemindex2]

itemindex = np.where(Y_range >= lower_limit_y)
X_below2 = X_range[itemindex]
Y_below2 = Y_range[itemindex]
L1_below2 = L1_range[itemindex]
L2_below2 = L2_range[itemindex]
nl_below2 = nl_range[itemindex]
nm_below2 = nm_range[itemindex]
nh_below2 = nh_range[itemindex]
dl_below2 = dl_range[itemindex]
dm_below2 = dm_range[itemindex]
dh_below2 = dh_range[itemindex]

itemindex2 = np.where(Y_below2 <= upper_limit_y)
X_final = X_below2[itemindex2]
Y_final = Y_below2[itemindex2]
L1_final = L1_below2[itemindex2]
L2_final = L2_below2[itemindex2]
nl_final = nl_below2[itemindex2]
nm_final = nm_below2[itemindex2]
nh_final = nh_below2[itemindex2]
dl_final = dl_below2[itemindex2]
dm_final = dm_below2[itemindex2]
dh_final = dh_below2[itemindex2]

limited_resp = np.column_stack((X_final,Y_final, L1_final, L2_final, nl_final,
                                nm_final, nh_final, dl_final, dm_final,
                                dh_final))

# use / (2*np.pi) to convert to rad/s

lowd = round(np.average(limited_resp[:,7]), 8)
lowdmin = round(np.min(limited_resp[:,7]), 8)
lowdmax = round(np.max(limited_resp[:,7]), 8)
lowdper = round((np.abs(lowdmax - lowdmin) / ((lowdmax + lowdmin)/2)) * 100, 2)

lown = round(np.average(limited_resp[:,4])/ (2*np.pi), 8)
lowmin = round(np.min(limited_resp[:,4])/ (2*np.pi), 8)
lowmax = round(np.max(limited_resp[:,4])/ (2*np.pi), 8)
lowper = round((np.abs(lowmax - lowmin) / ((lowmax + lowmin)/2)) * 100, 2)

medd = round(np.average(limited_resp[:,8]), 8)
meddmin = round(np.min(limited_resp[:,8]), 8)
meddmax = round(np.max(limited_resp[:,8]), 8)
meddper = round((np.abs(meddmax - meddmin) / ((meddmax + meddmin)/2)) * 100, 2)

medn = round(np.average(limited_resp[:,5])/ (2*np.pi), 8)
medmin = round(np.min(limited_resp[:,5])/ (2*np.pi), 8)
medmax = round(np.max(limited_resp[:,5])/ (2*np.pi), 8)
medper = round((np.abs(medmax - medmin) / ((medmax + medmin)/2)) * 100, 2)

highd = round(np.average(limited_resp[:,9]), 8)
highdmin = round(np.min(limited_resp[:,9]), 8)
highdmax = round(np.max(limited_resp[:,9]), 8)
highdper = round((np.abs(highdmax - highdmin) / ((highdmax + highdmin)/2)) * 100, 2)

highn = round(np.average(limited_resp[:,6])/ (2*np.pi), 8)
highmin = round(np.min(limited_resp[:,6])/ (2*np.pi), 8)
highmax = round(np.max(limited_resp[:,6])/ (2*np.pi), 8)
highper = round((np.abs(highmax - highmin) / ((highmax + highmin)/2)) * 100, 2)

print('The average of the Low Nat Freq: {}'.format(lown))
print('The min of the Low Nat Freq: {}'.format(lowmin))
print('The max of the Low Nat Freq: {}'.format(lowmax))
print('The Percent Difference: {}\n'.format(lowper))

print('The average of the Medium Nat Freq: {}'.format(medn))
print('The min of the Medium Nat Freq: {}'.format(medmin))
print('The max of the Medium Nat Freq: {}'.format(medmax))
print('The Percent Difference: {}\n'.format(medper))

print('The average of the High Nat Freq: {}'.format(highn))
print('The min of the High Nat Freq: {}'.format(highmin))
print('The max of the High Nat Freq: {}'.format(highmax))
print('The Percent Difference: {}\n'.format(highper))

print('The average of the Low Damping: {}'.format(lowd))
print('The min of the Low Damping: {}'.format(lowdmin))
print('The max of the Low Damping: {}'.format(lowdmax))
print('The Percent Difference: {}\n'.format(lowdper))

print('The average of the Medium Damping: {}'.format(medd))
print('The min of the Medium Damping: {}'.format(meddmin))
print('The max of the Medium Damping: {}'.format(meddmax))
print('The Percent Difference: {}\n'.format(meddper))

print('The average of the High Damping: {}'.format(highd))
print('The min of the High Damping: {}'.format(highdmin))
print('The max of the High Damping: {}'.format(highdmax))
print('The Percent Difference: {}\n'.format(highdper))

sns.set(context="paper", font="serif")
sns.axes_style({'font.family': ['serif']})

plt.figure(0)
plt.subplot(2, 3, 1)
hea1 = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
                      ,XY_L1L2_natflmh_damplmh[:,4]/(2*np.pi)))
df1 = pd.DataFrame(hea1, columns=['X','Y','EL'])
Heats1 = df1.pivot("Y", "X", "EL")
ax1 = sns.heatmap(Heats1, xticklabels=False, yticklabels=False, annot=False, fmt="d",
            linewidths=0, cmap="inferno", cbar=False)
#             vmin=0,
#             vmax=6)
# plt.title('Low Mode Natural Frequency Across Workspace', y = 1.05)
cbar1 = ax1.figure.colorbar(ax1.collections[0])
cbar1.ax.set_title(r'Hz')
plt.title('Low Natural')
plt.xlabel(r'X Position (20m)')
plt.ylabel(r'Y Position (15m)')
# plt.savefig("Low_Mode_Natural_Frequency.pdf")
# print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()

##############################################################################

# plt.figure(1)
plt.subplot(2, 3, 2)
hea2 = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
                      ,XY_L1L2_natflmh_damplmh[:,5]/(2*np.pi)))
df2 = pd.DataFrame(hea2, columns=['X','Y','EL'])
Heats2 = df2.pivot("Y", "X", "EL")
ax2 = sns.heatmap(Heats2, xticklabels=False, yticklabels=False, annot=False, fmt="d",
            linewidths=0, cmap="inferno", cbar=False)
#             vmin=0,
#             vmax=6)
# plt.title('Middle Mode Natural Frequency Across Workspace')
cbar2 = ax2.figure.colorbar(ax2.collections[0])
cbar2.ax.set_title('Hz')
plt.title('Middle Natural')
plt.xlabel('X Position (20m)')
plt.ylabel('Y Position (15m)')
# plt.savefig("Middle_Mode_Natural_Frequency.pdf")
# print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()

##############################################################################

# plt.figure(2)
plt.subplot(2, 3, 3)
hea3 = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
                      ,XY_L1L2_natflmh_damplmh[:,6]/(2*np.pi)))
df3 = pd.DataFrame(hea3, columns=['X','Y','EL'])
Heats3 = df3.pivot("Y", "X", "EL")
ax3 = sns.heatmap(Heats3, xticklabels=False, yticklabels=False, annot=False, fmt="d",
            linewidths=0, cmap="inferno", cbar=False)
#             vmin=0,
#             vmax=6)
# plt.title('Middle Mode Natural Frequency Across Workspace')
cbar3 = ax3.figure.colorbar(ax3.collections[0])
cbar3.ax.set_title('Hz')
plt.title('High Natural')
plt.xlabel('X Position (20m)')
plt.ylabel('Y Position (15m)')
# plt.savefig("High_Mode_Natural_Frequency.pdf")
# print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()

##############################################################################

# plt.figure(3)
plt.subplot(2, 3, 4)
hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
                      ,XY_L1L2_natflmh_damplmh[:,7]))
df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
            linewidths=0, cmap="inferno")
            # vmin=0,
            # vmax=2)
# plt.title('Low Mode Damping Across Workspace', y = 1.05)
plt.title('Low Damping')
plt.xlabel('X Position (20m)')
plt.ylabel('Y Position (15m)')
# plt.savefig("Low_Mode_Damping.pdf")
# print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()

##############################################################################

# plt.figure(4)
plt.subplot(2, 3, 5)
hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
                      ,XY_L1L2_natflmh_damplmh[:,8]))
df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
            linewidths=0, cmap="inferno")
            # vmin=0,
            # vmax=2)
# plt.title('Middle Mode Damping Across Workspace', y = 1.05)
plt.title('Middle Damping')
plt.xlabel('X Position (20m)')
plt.ylabel('Y Position (15m)')
# plt.savefig("Middle_Mode_Damping.pdf")
# print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()

##############################################################################

# plt.figure(5)
plt.subplot(2, 3, 6)
hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
                      ,XY_L1L2_natflmh_damplmh[:,9]))
df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
            linewidths=0, cmap="inferno")
            # vmin=0,
            # vmax=2)
# plt.title('High Mode Damping Across Workspace', y = 1.05)
plt.title('High Damping')
plt.xlabel('X Position (20m)')
plt.ylabel('Y Position (15m)')
# plt.savefig("High_Mode_Damping.pdf")
# print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
plt.tight_layout()
plt.savefig("Heatmaps_Subplots.pdf")
plt.show()
