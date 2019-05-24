#! /usr/bin/env python

###############################################################################
# forrest_2mode_sensplot.py
#
# Demonstrating the approximation of the 2-mode sensplot
#
# NOTE: Any plotting is set up for output, not viewing on screen.
#       So, it will likely be ugly on screen. The saved PDFs should look
#       better.
#
# Created: 04/12/16
#   - Joshua Vaughan
#   - joshua.vaughan@louisiana.edu
#   - http://www.ucs.louisiana.edu/~jev9637
#
# Modified:
#   *
#
###############################################################################

import numpy as np
import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

import InputShaping as shaping

zv = shaping.ZV(0.25, 0.017)
ei = shaping.EI(0.45, 0.05)

freq1, mag1 = shaping.sensplot(zv.shaper, 0, 1, 0.017)
freq2, mag2 = shaping.sensplot(ei.shaper, 0, 1, 0.05)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units font
plt.setp(ax.get_ymajorticklabels(),fontsize=18)
plt.setp(ax.get_xmajorticklabels(),fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Frequency (Hz)', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Percentage Vibration', fontsize=22, weight='bold', labelpad=10)

# plt.plot(freq1, mag1*mag2*100, linewidth=3, linestyle='-', label=r'ZV-EI')
# plt.plot(freq1, mag1*100, linewidth=3, linestyle='-.', label=r'ZV')

plt.plot([0, 0.9], [5,5], linewidth = 2, linestyle = '--')
plt.text(0.91, 2, r'$V_{tol}$', fontsize=15)
plt.plot(freq2, mag2*100, linewidth=3, linestyle='-', label=r'EI')


# uncomment below and set limits if needed
# plt.xlim(0,5)
plt.ylim(0,60)

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 3, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,fontsize=15)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
plt.savefig('EIsensplot.pdf')

# show the figure
plt.show()
