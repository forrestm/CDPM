
import numpy as np
from numpy import sin, cos, sqrt

# import pandas as pd
import sympy
import sympy.physics.mechanics as me
from decimal import Decimal
from scipy.linalg import eigvals

# import seaborn as sns
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt

# Edited April 9 to make more realistic cable K and C and mass. More reflective
# of the HiBot BRIDGEVIEW mass

# This is to calculate the Lengths of the cables
# for the Linearized Natural Frequency

# Create the variables
x, y, beta = me.dynamicsymbols('x, y, beta')

# Create the velocities
x_dot, y_dot, beta_dot = me.dynamicsymbols('x, y, beta', 1)

# Create the symbols
m, k, L, g, H, c, D,t = sympy.symbols('m k L g H c D t')
Izz, k_beta, c_beta = sympy.symbols('Izz k_beta c_beta')
L_1_init, L_2_init, Le1, Le2 = sympy.symbols('L_1_init L_2_init Le1 Le2')

# Create the world frame
N = me.ReferenceFrame('N')

# Create the rod frame
B = N.orientnew('B', 'axis', [beta, N.z])

# Set the rotation of the rod frame
B.set_ang_vel(N, -beta_dot * N.z)

# Create the Origin
O1 = me.Point('O_1')

# Set origin velocity to zero
O1.set_vel(N, 0 * N.x)

# Create the second attachment point
O2 = O1.locatenew('O_2', H * N.x)
O2.set_vel(N, 0 * N.x)

# Locate the point in the N frame
# P = me.Point('pen')
# P = O1.locatenew('P', x * N.x + y * N.y)
P = me.Point('P')
P.set_pos(O1, x * N.x + y * N.y)
P.set_pos(O2, -(H - x) * N.x + y * N.y)

# P.set_pos(O1, x * N.x + y * N.y)

# Set the point's velocity
P.set_vel(N, x_dot * N.x + y_dot * N.y)

# Create the rod center of mass
G = P.locatenew('G', D/2 * B.y)

# Set the velocity of G
G.v2pt_theory(P, N, B)

# Create the rod
I_rod = me.inertia(B, 0, 0, Izz)
rod = me.RigidBody('rod', G, B, m, (I_rod, G))

# Create the distance from the point to each attachment point
L1 = O1.pos_from(P).magnitude
L2 = O2.pos_from(P).magnitude
L1_vector = O1.pos_from(P).normalize
L2_vector = O2.pos_from(P).normalize

# F1 = (Le1 - L1()) * k * L1_vector()
# F2 = (Le2 - L2()) * k * L2_vector()

# Create the height from the center of gravity to the datum
h = G.pos_from(O1) & N.y

# The forces at the connection point
forceP = c * (x_dot + y_dot) * L1_vector() + c * (-x_dot + y_dot) * L2_vector()

# The forces on the beta frame
forceB = c_beta * beta_dot * N.z

rod.potential_energy = (-m * g * h + 0.5 * k * (L1() - L_1_init)**2 + 0.5 *
                        k *(L2() - L_2_init)**2 + 0.5 * k_beta * beta**2)

Lag = me.Lagrangian(N, rod)

LM = me.LagrangesMethod(Lag, [x, y, beta], forcelist=[(P, forceP),
                        (B, forceB)], frame=N)

EqMotion = LM.form_lagranges_equations()
lrhs = LM.rhs()

################################################################################
'''
This will create the x,y locations of each point in the workspace.
The workspace is 19 by 14 meters to offset the 20 by 15 area by 0.5 meters.
The points go along the y-axis top to bottom then move over and repeat.
This will return a 10000 by 4 numpy array that has the X and Y points,
and Length1 and Length2.

'''

def Lengths(x,y):
    k=10
    h=20
    m=1.0
    Fy = np.array([[k*y/np.sqrt(x**2 + y**2), k*y/np.sqrt(y**2 + (h - x)**2)],
                   [k*x/np.sqrt(x**2 + y**2), -h*k/np.sqrt(y**2 + (h - x)**2
                    )+k*x/np.sqrt(y**2 + (h - x)**2)]])
    a = np.array([2*k*y - 9.81*m,-h*k + 2*k*x])
    x = np.linalg.solve(Fy,a)
    return x

X_and_Y_points = 100

X = np.linspace(1.0, 19.0, num=X_and_Y_points)
Y = np.linspace(1.0, 14.0, num=X_and_Y_points)
n = X_and_Y_points
dims=np.array((0,0))
for i in range(n):
    x_i = np.repeat(X[i],n)
    total = np.column_stack((x_i,Y))
    dims = np.vstack((dims,total))
dims = dims[1:]

n = np.shape(dims)[0]
Lengths_to_dims = np.array((0,0))
for i in range(n):
    Length1, Length2 = Lengths(dims[i,0],dims[i,1])
    Lengths_only_holder = np.column_stack((Length1, Length2))
    Lengths_to_dims = np.vstack((Lengths_to_dims, Lengths_only_holder))
Lengths_to_dims = Lengths_to_dims[1:]
X_Y_L1_L2 = np.hstack((dims, Lengths_to_dims))
################################################################################


nat_freq_to_total = np.array((0,0,0))
damp_to_total = np.array((0,0,0))

# This will be done outside of the for loop to speed up computation

linearizer = LM.to_linearizer([x, y, beta], [x_dot, y_dot, beta_dot])
M, A, B = linearizer.linearize()

for i in range(n):

    op_point = {x:X_Y_L1_L2[i,0], y:X_Y_L1_L2[i,1], beta:0,
                x_dot:0, y_dot:0, beta_dot:0}

    constants = {m : 10.0,
              g : 9.81,
              k : 100.0,
              L_1_init:X_Y_L1_L2[i,2],
              L_2_init:X_Y_L1_L2[i,3],
              H : 20.0,
              c : 10.0,
              D : 3.0,
              Izz: 3**2 * (1.0/3.0) * 10,
              k_beta: 1.0,
              c_beta: 1.0}

    M_op = me.msubs(M, op_point)
    A_op = me.msubs(A, op_point)
    # B_op = me.msubs(B, op_point)
    perm_mat = linearizer.perm_mat
    A_lin = perm_mat.T * M_op.LUsolve(A_op)
    A_lin_constants = me.msubs(A_lin, constants)
    A_sol = A_lin_constants.subs(op_point).doit()

    A_np = np.array(np.array(A_sol), np.float)

    eigenvals, eigenvects = np.linalg.eig(A_np)

    eigen = eigenvals[0:5:2]
    eigen_abs = np.abs(eigen)

    damp = np.abs(np.real(eigen)/eigen_abs)
    damp_index = np.argsort(damp)
    highd, middled, lowd = damp[damp_index][::-1][:3][0:3]
    # print('The fundamental damp is: {}'.format(lowd))
    # print('The second damp is: {}'.format(middled))
    # print('The third damp is: {}'.format(highd))

    eigen_index = np.argsort(eigen_abs)
    high, middle, low = eigen_abs[eigen_index][::-1][:3][0:3]
    # print('The fundamental frequency is: {}'.format(low))
    # print('The second frequency is: {}'.format(middle))
    # print('The third frequency is: {}'.format(high))

    print(i)

    nat_freq_columns = np.column_stack((low,middle,high))
    nat_freq_to_total = np.vstack((nat_freq_to_total, nat_freq_columns))

    damp_columns = np.column_stack((lowd,middled,highd))
    damp_to_total = np.vstack((damp_to_total, damp_columns))

nat_freq_to_total = nat_freq_to_total[1:]
damp_to_total = damp_to_total[1:]
XY_L1L2_natflmh_damplmh = np.hstack((X_Y_L1_L2,nat_freq_to_total,damp_to_total))


np.savetxt("XY_L1L2_natflmh_damplmh2.csv", XY_L1L2_natflmh_damplmh,delimiter=",")


# plt.figure(0)
# hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
#                       ,XY_L1L2_natflmh_damplmh[:,4]))
# df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
# Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
# sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
#             linewidths=0, cmap="inferno")
#             # vmin=0,
#             # vmax=2)
# plt.title('Low Mode Natural Frequency Across Workspace', y = 1.05)
# plt.savefig("Low_Mode_Natural_Frequency.pdf")
# # print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()
#
# plt.figure(1)
# hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
#                       ,XY_L1L2_natflmh_damplmh[:,5]))
# df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
# Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
# sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
#             linewidths=0, cmap="inferno")
#             # vmin=0,
#             # vmax=2)
# plt.title('Middle Mode Natural Frequency Across Workspace', y = 1.05)
# plt.savefig("Middle_Mode_Natural_Frequency.pdf")
# # print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()
#
# plt.figure(2)
# hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
#                       ,XY_L1L2_natflmh_damplmh[:,6]))
# df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
# Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
# sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
#             linewidths=0, cmap="inferno")
#             # vmin=0,
#             # vmax=2)
# plt.title('High Mode Natural Frequency Across Workspace', y = 1.05)
# plt.savefig("High_Mode_Natural_Frequency.pdf")
# # print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()
#
# plt.figure(3)
# hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
#                       ,XY_L1L2_natflmh_damplmh[:,7]))
# df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
# Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
# sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
#             linewidths=0, cmap="inferno")
#             # vmin=0,
#             # vmax=2)
# plt.title('Low Mode Damping Across Workspace', y = 1.05)
# plt.savefig("Low_Mode_Damping.pdf")
# # print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()
#
# plt.figure(4)
# hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
#                       ,XY_L1L2_natflmh_damplmh[:,8]))
# df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
# Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
# sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
#             linewidths=0, cmap="inferno")
#             # vmin=0,
#             # vmax=2)
# plt.title('Middle Mode Damping Across Workspace', y = 1.05)
# plt.savefig("Middle_Mode_Damping.pdf")
# # print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()
#
# plt.figure(5)
# hea = np.column_stack((XY_L1L2_natflmh_damplmh[:,0],XY_L1L2_natflmh_damplmh[:,1]
#                       ,XY_L1L2_natflmh_damplmh[:,9]))
# df4 = pd.DataFrame(hea, columns=['X (meters)','Y (meters)','EL'])
# Heats5 = df4.pivot("Y (meters)", "X (meters)", "EL")
# sns.heatmap(Heats5, xticklabels=False, yticklabels=False, annot=False, fmt="d",
#             linewidths=0, cmap="inferno")
#             # vmin=0,
#             # vmax=2)
# plt.title('High Mode Damping Across Workspace', y = 1.05)
# plt.savefig("High_Mode_Damping.pdf")
# # print("The Low average is: {}".format(np.average(nat_freq_high_eigen_no_zero)))
# plt.show()
