#!/usr/bin/python
###############################################################################
# Filename    : Natural_Freq_Eigen_Calcs_v2.py
# Created     : 2016-05-26
# Author      : Forrest
'''
Description   :
This file will calculate the natural frequencies and damping ratios of the CDPM.
It will then export the data to a csv file which can be used to create heatmaps
of the workspace.
'''
# Modified    :
###############################################################################


import numpy as np
from numpy import sin, cos, sqrt

import pandas as pd
import sympy
import sympy.physics.mechanics as me
from decimal import Decimal
from scipy.linalg import eigvals

import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt

workspace_width = 20
left_cable_k = 100
right_cable_k = 100
rod_k = 250.0
cable_c = 10.0
rod_c = 10.0
plate_width = 4.0
plate_height = 2.0
mass_of_plate = 10.0
rod_length = 3.0
mass_of_rod = 2.0
rod_init = (9.81 * mass_of_rod) / rod_k
inertia_of_plate = (plate_width**2 + plate_height**2) * (mass_of_plate/12.0)
inertia_of_rod = (mass_of_rod * rod_length**2)/12.0

# Create the frames
N = me.ReferenceFrame('N')
B = me.ReferenceFrame('B')

# Create the symbols
x, y, beta, e, F = me.dynamicsymbols('x y beta e F')
x_dot, y_dot, beta_dot, e_dot = me.dynamicsymbols('x_dot y_dot beta_dot e_dot')
H, a, b, m, g, k, t, L1, L2 = sympy.symbols('H a b m g k t L1 L2')
c, c_rod, D, M, k_rod, mom = sympy.symbols('c c_rod D M k_rod mom')
Izz, Izz_rod = sympy.symbols('Izz Izz_rod')

# Orient the Beta frame
B.orient(N, 'Axis', (beta, N.z))
B.set_ang_vel(N, beta_dot * N.z)

# Create the first point
O1 = me.Point('O1')
O2 = me.Point('O2')
O1.set_pos(O1, 0)
O2.set_pos(O1, H * N.x)

G = me.Point('G')
G.set_pos(O1, x*N.x - y*N.y)

P1 = me.Point('P1')
P1.set_pos(G, -a/2 * B.x + b/2 * B.y)
P2 = me.Point('P2')
P2.set_pos(G, a/2 * B.x + b/2 * B.y)

O1.set_vel(N, 0)
O2.set_vel(N, 0)
G.set_vel(B, 0)
G.set_vel(N, x_dot * N.x - y_dot * N.y)
P1.v2pt_theory(G,N,B)
P2.v2pt_theory(G,N,B)
P1.a2pt_theory(G,N,B)
P2.a2pt_theory(G,N,B)

Z_G = G.locatenew('Z_G', e * B.y)
Z_G.set_vel(B, e_dot * B.y)
Z_G.v1pt_theory(G,N,B)
Z_G.a1pt_theory(G,N,B)

kde = [x_dot - x.diff(t), y_dot - y.diff(t), beta_dot - beta.diff(t),
       e_dot - e.diff(t)]

I_plate = me.inertia(N, 0, 0, Izz)
inertia_plate = (I_plate, G)

I_rod = me.inertia(N, 0, 0, Izz_rod)
inertia_rod = (I_rod, Z_G)

Plate = me.RigidBody('Plate', G, B, M, inertia_plate)
rod = me.RigidBody('rod', Z_G, B, m, inertia_rod)


def Lengths_and_Moments(x,y):
    '''
    This function determines the initial length of the springs and the
    moment needed in order to keep the plate in position.
    '''
    xxx = x
    yyy = y
    a,b,x,y,H, k1, k2, Length1, Length2 = sympy.symbols('a b x y H k1 k2 Length1 Length2')
    m, Fsp1,Fsp2,Ma,t1,t2 = sympy.symbols('m Fsp1 Fsp2 Ma t1 t2')
    L1 = sympy.sqrt((x-(a/2))**2 + (y-(b/2))**2)
    L2 = sympy.sqrt((-(H-x)+(a/2))**2 + (y-(b/2))**2)

    # Calculating the Spring Forces
    Fsp1 = k1*(L1 - Length1)
    Fsp2 = k2*(L2 - Length2)

    # Calculating the left and right spring force's x component
    leftx = (x-(a/2))
    rightx = (H - leftx - a)

    # Calculating the left and right spring force's y component
    y_same = y-(b/2)

    # Calculating the left and right x component of the spring force
    Fx1 = Fsp1 * (leftx / L1)
    Fx2 = Fsp2 * (rightx / L2)

    # Setting up the total x equation
    Fx = Fx1 - Fx2

    # Calculating the left and right y component of the spring force
    Fy1 = Fsp1 * (y_same / L1)
    Fy2 = Fsp2 * (y_same / L2)

    # Setting up the total y equation
    Fy = Fy1 + Fy2 - 9.81*m

    M = Fy2 * a - 9.81*m*(a/2) + Ma
    Totalx = Fx1 - Fx2
    Totaly = Fy1 + Fy2 - 9.81*m

    Totalx = (Fx.subs({x:xxx, y:yyy, a:4.0,b:2.0, H:20.0, k1:100.0, 
                       k2:100.0})).evalf()
    Totaly = (Fy.subs({x:xxx, y:yyy, a:4.0,b:2.0, H:20.0, k1:100.0,
                       k2:100.0, m:12.0})).evalf()
    Moment = (M.subs( {x:xxx, y:yyy, a:4.0,b:2.0, H:20.0, k1:100.0,
                       k2:100.0, m:12.0})).evalf()
    abv = sympy.solve([sympy.Eq(Totalx, 0.0),
                       sympy.Eq(Totaly, 0.0),
                       sympy.Eq(Moment, 0.0)], [Length1, Length2, Ma])
    l1 = abv[Length1]
    l2 = abv[Length2]
    mo = abv[Ma]
    return l1,l2,mo

# Adding the forces of gravity on the plate and the rod
grav_force_plate = (G, -M * g * N.y)
grav_force_rod = (Z_G, -m * g * N.y)

# Calculating the Geometric lengths from the top corners of the plate
Length1 = P1.pos_from(O1).magnitude()
Length2 = P2.pos_from(O2).magnitude()

# Creating the unit vectors pointing from the origins to the top plate points
P1_vector = P1.pos_from(O1).normalize()
P2_vector = P2.pos_from(O2).normalize()

# These spring functions do not allow the springs to exhibit compression force
def K1(Length1,x,y):
    k = left_cable_k
    L = Lengths_and_Moments(x, y)[0]

    return k * (Length1 >= L)

def K2(Length2,x,y):
    k = right_cable_k
    L = Lengths_and_Moments(x, y)[1]

    return k * (Length2 >= L)

# The name of these variables is confusing this is simply the forces of the
# springs directed in the correct direction
spring_1_vector_P1 = -(P1.pos_from(O1).normalize()) * k * (Length1 - L1)
spring_2_vector_P2 = -(P2.pos_from(O2).normalize()) * k * (Length2 - L2)

# Storing the forces and respective points in tuple
spring_1_force_P1 = (P1, spring_1_vector_P1)
spring_2_force_P2 = (P2, spring_2_vector_P2)

# This is setting up the forces of the rod spring on the rod and on the plate
spring_force_rod = (Z_G, -k_rod*e*B.y)
spring_force_rod_on_plate = (G, k_rod*e*B.y)

# This is the cable damping forces acting on the plate
P1_damp = -(c * P1.vel(N) & P1_vector) * P1_vector
P2_damp = -(c * P2.vel(N) & P2_vector) * P2_vector

# Storing the forces and respective points in tuple
damping_1 = (P1, P1_damp)
damping_2 = (P2, P2_damp)

# These are the damping forces acting on the rod and on the plate from the rod
damping_rod = (Z_G, -c_rod * e_dot * B.y)
damping_rod_on_plate = (G, c_rod * e_dot * B.y)

# This is the moment that needs to be applied in order for the plate to stay in
# static equilibrium
moment = (B, Lengths_and_Moments(x,y)[2] * N.z)

# Setting up the coordinates speeds and creating the calling KanesMethod
coordinates = [x, y, beta, e]
speeds= [x_dot, y_dot, beta_dot,e_dot]
kane = me.KanesMethod(N, coordinates, speeds, kde)

loads = [spring_1_force_P1, spring_2_force_P2, grav_force_plate,grav_force_rod,
         spring_force_rod,spring_force_rod_on_plate, damping_rod,
        damping_rod_on_plate,damping_1,damping_2, moment]

fr, frstar = kane.kanes_equations(loads, [Plate, rod])

#Mass = kane.mass_matrix_full
#f = kane.forcing_full

################################################################################
'''
This will create the x,y locations of each point in the workspace.
The workspace is 19 by 14 meters to offset the 20 by 15 area by 0.5 meters.
The points go along the y-axis top to bottom then move over and repeat.
This will return a 10000 by 4 numpy array that has the X and Y points,
and Length1 and Length2.

'''

X_and_Y_points = 100

X = np.linspace(3.0, 18.0, num=X_and_Y_points)
Y = np.linspace(3.0, 14.0, num=X_and_Y_points)
n = X_and_Y_points
dims=np.array((0,0))
for i in range(n):
    x_i = np.repeat(X[i],n)
    total = np.column_stack((x_i,Y))
    dims = np.vstack((dims,total))
dims = dims[1:]

n = np.shape(dims)[0]
Lengths_to_dims = np.array((0,0,0))
for i in range(n):
    Length1 = 2.5e-5*(2943.0*dims[i,0] + 40000.0*dims[i,1] -
                      92974.0)*sqrt(dims[i,0]**2 - 4.0*dims[i,0] + dims[i,1]**2 -
                                    2.0*dims[i,1] + 5.0)/(dims[i,1] - 1.0)
    Length2 = (-0.073575*dims[i,0] + dims[i,1] - 0.85285)*sqrt(dims[i,0]**2 - 36.0*dims[i,0] +
                                                     dims[i,1]**2 - 2.0*dims[i,1] + 325.0)/(dims[i,1] - 1.0)
    Torque = -29.43*dims[i,0] + 294.3
#     Length1, Length2, Torque = Lengths_and_Moments(dims[i,0],dims[i,1])
    Lengths_only_holder = np.column_stack((Length1, Length2, Torque))
    Lengths_to_dims = np.vstack((Lengths_to_dims, Lengths_only_holder))
Lengths_to_dims = Lengths_to_dims[1:]
X_Y_L1_L2_T = np.hstack((dims, Lengths_to_dims))
################################################################################
nat_freq_to_total = np.array((0,0,0,0))
damp_to_total = np.array((0,0,0,0))

# This will be done outside of the for loop to speed up computation

linearizer = kane.to_linearizer()
Maz, A, B = linearizer.linearize()

for i in range(n):

    op_point = {x:X_Y_L1_L2_T[i,0], y:X_Y_L1_L2_T[i,1], beta:0, e:-0.07848,
            x_dot:0, y_dot:0, beta_dot:0, e_dot:0}

    constants = {m:2.0,
                 M:10,
                 g:9.81,
                 k:100,
                 H:20.0,
                 a:4.0,
                 b:2.0,
                 c:10.0,
                 c_rod:10.0,
                 k_rod:250.0,
                 Izz:16.666666666666668,
                 Izz_rod:1.5,
                 L1:X_Y_L1_L2_T[i,2],
                 L2:X_Y_L1_L2_T[i,3],
                }

    M_op = me.msubs(Maz, op_point)
    A_op = me.msubs(A, op_point)
    perm_mat = linearizer.perm_mat
    A_lin = perm_mat.T * M_op.LUsolve(A_op)
    A_lin_constants = me.msubs(A_lin, constants)
    A_sol = A_lin_constants.subs(op_point).doit()

    A_np = np.array(np.array(A_sol), np.float)

    eigenvals, eigenvects = np.linalg.eig(A_np)

    eigen = eigenvals[0:7:2]
    eigen_abs = np.abs(eigen)

    damp = np.abs(np.real(eigen)/eigen_abs)
    damp_index = np.argsort(damp)
    highd, middled, middled2, lowd = damp[damp_index][::-1][:4][0:4]
    # print('The fundamental damp is: {}'.format(lowd))
    # print('The second damp is: {}'.format(middled))
    # print('The third damp is: {}'.format(highd))

    eigen_index = np.argsort(eigen_abs)
    high, middle, middle2, low = eigen_abs[eigen_index][::-1][:4][0:4]
    # print('The fundamental frequency is: {}'.format(low))
    # print('The second frequency is: {}'.format(middle))
    # print('The third frequency is: {}'.format(high))

    print(i)

    nat_freq_columns = np.column_stack((low,middle,middle2,high))
    nat_freq_to_total = np.vstack((nat_freq_to_total, nat_freq_columns))

    damp_columns = np.column_stack((lowd,middled,middled2,highd))
    damp_to_total = np.vstack((damp_to_total, damp_columns))

nat_freq_to_total = nat_freq_to_total[1:]
damp_to_total = damp_to_total[1:]
XY_L1L2_natflm2h_damplm2h = np.hstack((X_Y_L1_L2_T,nat_freq_to_total,damp_to_total))


np.savetxt("XY_L1L2_natflm2h_damplm2h.csv", XY_L1L2_natflm2h_damplm2h,delimiter=",")

