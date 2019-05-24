#!/usr/bin/python
###############################################################################
# Filename    : CDPM_v2_Kanes_static.py
# Created     : 2016-11-2
# Author      : Forrest
'''
Description   :
This program will create the EOM for a CDPM with a horizontal plate attached to
a ceiling by two springs connected to its' two upper corners:

+----------------------------------> X
|XXX                            XXX
|  XX                          XX
|   XXX                       XX
|     XX                    XXX
|       XX                 XX
|        XXX--------------XX
|          |              |
|          |              |
|          |              |
↓          +--------------+
Y

The code will then plot the response given the x_init and y_init values. The
program will also animate the response provided you supply the CDPMv2_animation
file. This file does not move the payload.
'''
# Modified    :
###############################################################################

from sympy import symbols, init_printing
import sympy
import sympy.physics.mechanics as me
from pydy.system import System
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
import numpy as np
from CDPMv2_animation import animate as ani

# Put in your initial conditions
x_init = 8
y_init = 10

# You should probably leave beta at 0 initially
beta_init = 0

############################## Animation ####################################
# Do you want this to be animated?
want_animation = True

# What do you want animation filename to be?
anim_filename = "CDPM_Plate_static"

# How many seconds do you want the animation to be?
anim_sec = 10
#############################################################################

# The CDPM properties
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
Simulation_Time = 20

# Create the frames
N = me.ReferenceFrame('N')
B = me.ReferenceFrame('B')

# Create the symbols
x, y, beta, e, F, L1, L2 = me.dynamicsymbols('x y beta e F L1 L2')
x_dot, y_dot, beta_dot, e_dot = me.dynamicsymbols('x_dot y_dot beta_dot e_dot')
H, a, b, m, g, k, t = sympy.symbols('H a b m g k t')
c, c_rod, D, M, k_rod, mom = sympy.symbols('c c_rod D M k_rod mom')
Izz, Izz_rod = sympy.symbols('Izz Izz_rod')

# Orient the Beta frame
B.orient(N, 'Axis', (beta, N.z))
B.set_ang_vel(N, beta_dot * N.z)

# Create the first and second point
O1 = me.Point('O1')
O2 = me.Point('O2')
O1.set_pos(O1, 0)
O2.set_pos(O1, H * N.x)

# Create the plate center of mass
G = me.Point('G')
G.set_pos(O1, x*N.x - y*N.y)

# Create the cable attachment points
P1 = me.Point('P1')
P1.set_pos(G, -a/2 * B.x + b/2 * B.y)
P2 = me.Point('P2')
P2.set_pos(G, a/2 * B.x + b/2 * B.y)

# Set all points velocities
O1.set_vel(N, 0)
O2.set_vel(N, 0)
G.set_vel(B, 0)
G.set_vel(N, x_dot * N.x - y_dot * N.y)
P1.v2pt_theory(G,N,B)
P2.v2pt_theory(G,N,B)
P1.a2pt_theory(G,N,B)
P2.a2pt_theory(G,N,B)

# Create rod center of mass
Z_G = G.locatenew('Z_G', e * B.y)
Z_G.set_vel(B, e_dot * B.y)
Z_G.v1pt_theory(G,N,B)
Z_G.a1pt_theory(G,N,B)

# Create the kinematic equations
kde = [x_dot - x.diff(t), y_dot - y.diff(t), beta_dot - beta.diff(t),
       e_dot - e.diff(t)]

# Define the inertial tensors
I_plate = me.inertia(N, 0, 0, Izz)
inertia_plate = (I_plate, G)

I_rod = me.inertia(N, 0, 0, Izz_rod)
inertia_rod = (I_rod, Z_G)

# Create the rigidbodies
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

# This is an approximation of what the cable lengths should be
L1_init = Lengths_and_Moments(x_init,y_init)[0]
L2_init = Lengths_and_Moments(x_init,y_init)[1]

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
spring_1_vector_P1 = -(P1.pos_from(O1).normalize()) * K1(Length1,x,y) * (Length1 - L1)
spring_2_vector_P2 = -(P2.pos_from(O2).normalize()) * K2(Length2,x,y) * (Length2 - L2)

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

Mass = kane.mass_matrix_full
f = kane.forcing_full

sys = System(kane)
sys.constants = {m:mass_of_rod,
                 M:mass_of_plate,
                 g:9.81,
                 H:workspace_width,
                 a:plate_width,
                 b:plate_height,
                 c:cable_c,
                 c_rod:rod_c,
                 k_rod:rod_k,
                 Izz:inertia_of_plate,
                 Izz_rod:inertia_of_rod,
                 }
sys.initial_conditions = {x:x_init, y:y_init, beta:beta_init, e:-rod_init}
sys.times = np.linspace(0.0, Simulation_Time, 1000)
sys.generate_ode_function(generator='cython')

y = sys.integrate()

sim_time = np.linspace(0.0, Simulation_Time, 1000)
fig = plt.figure(figsize=(18, 4))

# fig = plt.figure(0)
fig.add_subplot(141)
plt.plot(sim_time, y[:,0], label='Unshaped')
# plt.ylim(25,0)
plt.title(r'X Motion')

fig.add_subplot(142)
plt.plot(sim_time, y[:,1], label='Unshaped')
plt.gca().invert_yaxis()
# plt.ylim(20,0)
plt.title(r'Y Motion')

fig.add_subplot(143)
plt.plot(sim_time, np.degrees(y[:,2]), label='Unshaped')
plt.title(r'$\beta$ Motion')

fig.add_subplot(144)
plt.plot(sim_time, y[:,3], label='Unshaped')
# plt.ylim(20,0)
plt.title(r'Rod Motion')

# fig.add_subplot(224)
# plt.plot(y[:,0], y[:,1], label='Unshaped')
# # plt.gca().invert_yaxis()
# plt.xlim(0,20)
# plt.ylim(20,0)
# plt.title(r'Y Motion')

# plt.tight_layout()
plt.show()
if want_animation:
    ani(y,plate_width,plate_height,rod_length,anim_sec,anim_filename)
