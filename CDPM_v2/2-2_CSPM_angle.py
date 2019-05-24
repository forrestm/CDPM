#!/usr/bin/python
###############################################################################
# Filename    : 2-2_CSPM_angle.py
# Created     : March 3, 2017
# Author      : Forrest
'''
Description   :
This file takes coordinates for a 2-2 CSPM and outputs the angles and cable lengths
of the platform. These are just close guesses.
'''
# Modified    :
###############################################################################
# coding: utf-8

# In[1]:

from sympy import symbols, init_printing
import sympy
from sympy.utilities.lambdify import lambdastr
import sympy.physics.mechanics as me
from pydy.system import System
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
# get_ipython().magic('matplotlib inline')


# In[2]:

# Things to ask for
e_offset_scalar = 1.605
workspace_width = 10.4
cable_k = 1000
cable_c = 50
spring_on_frame_C = 10000

plate_width = 0.5
plate_height = 0.3
plate_mass = 10

runtime = 30

rod_radius   = 0.02
rod_length   = 2.91
rod_mass = 2
rod_spring = 1000
rod_damper = 20

plate_inertia = (plate_width**2 + plate_height**2) * (plate_mass/12.0)
rod_inertia = (rod_mass/12)*(3*rod_radius**2 + rod_length**2)


# Move the workspace in 1 meter 
x_begin = 1
x_end = workspace_width - 1

# Move the workspace down 0.5 meters
z_begin = 0.5
z_end = 3


# In[3]:

# Create the symbols
x_dot, z_dot, e_dot = sympy.symbols('x_dot z_dot e_dot')
theta_dot, phi_dot = sympy.symbols('theta_dot phi_dot')
H, a, b, M, m, g, k, t = sympy.symbols('H a b M m g k_{cable} t')
Ip, Ir, c, r, p, kr, cr, D = sympy.symbols('I_{plate} I_{rod} c r p k_{rod} c_{rod} D')
L1, L2, e_offset, k_C = sympy.symbols('L1 L2 e_{offset} k_{CFrame}')

# Create the frames Z+ points down, and Y+ is out of the screen
A = me.ReferenceFrame('A')

for j in range(2):
    if j == 0:
        x, z, e, theta, phi = sympy.symbols('x z e theta phi')
        # Orient the Beta frame
        B = A.orientnew('B', 'Axis', [theta, A.y])
        C = A.orientnew('C', 'Axis', [phi, A.y])
        B.set_ang_vel(A, theta_dot * A.y)
        C.set_ang_vel(A, phi_dot * A.y)
    else:
        x, z, e, theta, phi = me.dynamicsymbols('x z e theta phi')
        # Orient the Beta frame
        B = A.orientnew('B', 'Axis', [theta, A.y])
        C = A.orientnew('C', 'Axis', [phi, A.y])
        B.set_ang_vel(A, theta_dot * A.y)
        C.set_ang_vel(A, phi_dot * A.y)

    # Create the origin points
    A1 = me.Point('A1')
    A2 = me.Point('A2')

    # Set the origin points positions
    A1.set_pos(A1, 0)
    A2.set_pos(A1, H * A.x)

    # Create the plate and rod center
    G = me.Point('G')
    Gr = me.Point('Gr')

    # Set both centers position
    G.set_pos(A1, x*A.x + z*A.z)
    Gr.set_pos(G, e * C.z)

    # Create the attachment points
    B1 = me.Point('B1')
    B2 = me.Point('B2')

    # # Set the attachment points positions
    B1.set_pos(G, -a/2 * B.x - b/2 * B.z)
    B2.set_pos(G, a/2 * B.x - b/2 * B.z)

    # Create Rod top and Bottom points
    C1 = me.Point('C1')
    C2 = me.Point('C2')
    C1.set_pos(Gr, -D/2 * C.z)
    C2.set_pos(Gr, D/2 * C.z)

    # Create the position vectors
    a2 = A2.pos_from(A1)
    a2_x = a2 & A.x
    a2_z = a2 & A.z

    r21 = B2.pos_from(B1)
    r21_x = r21 & A.x
    r21_z = r21 & A.z

    s1 = B1.pos_from(A1)
    s2 = B2.pos_from(A2)

    spF1 = A1.pos_from(B1)
    spF2 = A2.pos_from(B2)

    s_rod = Gr.pos_from(G)

    # Calculating the Geometric lengths from the top corners of the plate
    Length1 = s1.magnitude()
    Length2 = s2.magnitude()

    # Creating the unit vectors pointing from the origins to the top plate points
    s1_vector = s1.normalize()
    s2_vector = s2.normalize()

    # Calculate the distance from the rigidbodies centers to datum
    rod_center_distance = A1.pos_from(Gr) & A.z
    plate_center_distance = A1.pos_from(G) & A.z

    # Set velocity of origin points
    A1.set_vel(A, 0)
    A2.set_vel(A, 0)

    # Set velocity of COG
    G.set_vel(A, x_dot * A.x + z_dot * A.z)
    G.set_vel(B, 0)

    Gr.set_vel(C, e_dot * C.z)
    Gr.v1pt_theory(G, A, C)

    # Set velocity of attachment points
    B1.v2pt_theory(G, A, B)
    B2.v2pt_theory(G, A, B)
    B1.set_vel(B,0)
    B2.set_vel(B,0)

    # Calculate the center of mass from the origin
    rod_mass_x = Gr.pos_from(A1) & A.x
    rod_mass_z = Gr.pos_from(A1) & A.z

    plate_mass_x = G.pos_from(A1) & A.x
    plate_mass_z = G.pos_from(A1) & A.z

    COM_x = (((rod_mass_x * rod_mass) + 
             (plate_mass_x * plate_mass)) / (plate_mass + rod_mass))

    COM_z = (((rod_mass_z * rod_mass) + 
             (plate_mass_z * plate_mass)) / (plate_mass + rod_mass))

    COM = A1.locatenew('COM', COM_x * A.x + COM_z * A.z)

    r1 = B1.pos_from(COM)
    r1_x = r1 & A.x
    r1_z = r1 & A.z

    r2 = B2.pos_from(COM)
    r2_x = r2 & A.x
    r2_z = r2 & A.z

    e_equil = -((-(9.81 * rod_mass) / rod_spring) + e_offset_scalar)

    if j == 0:
        p = ((r21_x * x + a2_x * r1_x) * z - r21_z * x**2 +
             (r1_z * r2_x - r1_x * r2_z + a2_x * r21_z - a2_z * r2_x) * x +
             r1_x * (a2_x * r2_z - a2_z * r2_x))
        equation_all = p.subs({H:workspace_width, a:plate_width, 
                               b:plate_height, phi:theta, e:e_equil}).evalf()
        equation_all = equation_all.simplify()
        lambda_str1 = lambdastr((theta), equation_all)
        lambda_str2 = lambda_str1.replace('sin', 'np.sin')
        lambda_str3 = lambda_str2.replace('cos', 'np.cos')
        lambda_str4 = lambda_str3.replace('x', 'x_temp')
        lambda_str = lambda_str4.replace('z', 'z_temp')
        func1 = eval(lambda_str)


# In[4]:

x_values_heat = np.linspace(x_begin, x_end, 100)
z_values_heat = np.linspace(z_begin, z_end, 100)

theta_values_heat = []
x_values_heatp = []
z_values_heatp = []
for j in range(100):
    z_temp = z_values_heat[j]
    for i in range(100):
        x_temp = x_values_heat[i]
        tau_initial_guess = 0.01
        tau_solution = fsolve(func1, tau_initial_guess)
        theta_values_heat.append(tau_solution)
        x_values_heatp.append(x_temp)
        z_values_heatp.append(z_temp)
x_values_heat_np = np.asarray(x_values_heatp)
z_values_heat_np = np.asarray(z_values_heatp)
theta_values_heat_np = np.asarray(theta_values_heat)
theta_map = np.rad2deg(theta_values_heat_np.reshape(100,100))


# In[5]:

# title = 'Angle Over Workspace'
# fig, ax = plt.subplots()
# x = x_values_heat
# y = z_values_heat
# X,Y = np.meshgrid(x,y)

# plt.pcolormesh(X,Y,theta_map, cmap=plt.cm.RdYlGn)
# ax.invert_yaxis()
# ax.axis('image')
# ax.set_aspect('equal')
# cbar = plt.colorbar()
# cbar.ax.set_title('Angle')
# plt.xlabel('Horizontal Motion')
# plt.ylabel('Vertical Motion')
# plt.title(title)
# plt.savefig('/Users/forrest/Desktop/angles_COF_2plate' + ".pdf")


# In[6]:

Length1_values_pre_numpy = []
Length2_values_pre_numpy = []
for i in range(10000):
    Length1_temp = Length1.subs({   M: plate_mass,
                     m: rod_mass,
                     g: 9.81,
                     H: workspace_width,
                     a: plate_width,
                     b: plate_height,
                      x:x_values_heat_np[i],
                      z:z_values_heat_np[i],
                  theta:theta_values_heat_np[i],
                      e:e_equil, 
                    phi:theta_values_heat_np[i]
                     }).evalf()
    Length1_values_pre_numpy.append(Length1_temp)
    Length2_temp = Length2.subs({   M: plate_mass,
                     m: rod_mass,
                     g: 9.81,
                     H: workspace_width,
                     a: plate_width,
                     b: plate_height,
                      x:x_values_heat_np[i],
                      z:z_values_heat_np[i],
                  theta:theta_values_heat_np[i],
                      e:e_equil, 
                    phi:theta_values_heat_np[i]
                     }).evalf()
    Length2_values_pre_numpy.append(Length2_temp)
Length1_values = np.asarray(Length1_values_pre_numpy)
Length2_values = np.asarray(Length2_values_pre_numpy)


# In[ ]:

np.save('Length1_values', Length1_values)
np.save('Length2_values', Length2_values)
np.save('theta_values_heat_np', theta_values_heat_np)
np.save('x_values_heat_np', x_values_heat_np)
np.save('z_values_heat_np', z_values_heat_np)




