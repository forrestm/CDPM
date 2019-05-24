import matplotlib.pyplot as plt
import numpy as np
whole = np.genfromtxt ('XY_L1L2_natflm2h_damplm2h.csv', delimiter=",")

title = 'Damp Low'
fig, ax = plt.subplots()
x = whole[:,0]
y = whole[:,1]
z = whole[:,9]/(2*np.pi)
x=np.unique(x)
y=np.unique(y)
Y,X = np.meshgrid(y,x)

Z=z.reshape(len(y),len(x))

plt.pcolormesh(X,Y,Z, cmap=plt.cm.Blues)
ax.invert_yaxis()
ax.axis('image')

cbar = plt.colorbar()
cbar.ax.set_title('Hz')
plt.xlabel('Horizontal Motion')
plt.ylabel('Vertical Motion')
plt.title(title)
plt.savefig(title + ".pdf")
