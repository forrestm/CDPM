#!/usr/bin/python
###############################################################################
# Filename    : CDPM_Moment_Proof.py
# Created     : 2016-05-23
# Author      : Forrest
'''
Description   :
This is a sympy proof outlining that sympy does not actively solve reaction
forces. This necessitates the manual solving and supply of these often "hidden"
forces. The proof was created as to relive my stress about manually adding a
moment to the theta frame, which I initially thought was not organic and not the
best solution.
The idea is that for a rectagular 2D box the forces opposing the force of
gravity are directed at the two upper corners of the workspace:
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

When the box is centered along the X axis there only exists the forces. This is
fine for sympy. However when the box is located off center a combination of
forces and moments are present. These moments must be accounted for and this
file proves that for an off centered box the moments must be manually added.

Give initial x an off center value (above or below 10) and watch as the box
moves as it is not given a moment and static equilibrium is not achieved.

Pydy is needed to run this code use "pip install pydy" or
                                    "conda install -c pydy pydy"
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

inital_x = 13
inital_y = 10

N = me.ReferenceFrame('N')
B = me.ReferenceFrame('B')

x, y, beta, e, F,K = me.dynamicsymbols('x y beta e F K')
x_dot, y_dot, beta_dot, e_dot = me.dynamicsymbols('x_dot y_dot beta_dot e_dot')
H, a, b, m, g, L1, L2, k, t, c, D, M, k_rod, mom = sympy.symbols('H a b m g L1 L2 k t c D M k_rod mom')
Izz, Izz_rod, F1, F2 = sympy.symbols('Izz Izz_rod F1 F2')

B.orient(N, 'Axis', (beta, N.z))
B.set_ang_vel(N, beta_dot * N.z)

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

kde = [x_dot - x.diff(t), y_dot - y.diff(t), beta_dot - beta.diff(t)]

def Lengths_and_Moments(x,y):
    xxx = x
    yyy = y
    a,b,x,y,H, k1, k2, Length1, Length2 = sympy.symbols('a b x y H k1 k2 Length1 Length2')
    m, Fsp1,Fsp2,Ma,t1,t2 = sympy.symbols('m Fsp1 Fsp2 Ma t1 t2')
    L1 = sympy.sqrt((x-(a/2))**2 + (y-(b/2))**2)
    L2 = sympy.sqrt((-(H-x)+(a/2))**2 + (y-(b/2))**2)

    # Calculating the Spring Forces
#     Fsp1 = k1*(L1 - Length1)
#     Fsp2 = k2*(L2 - Length2)

    # Calculating the left and right spring force's x component
    leftx = (x-(a/2))
    rightx = (H - leftx - a) 

    # Calculating the y direction for the force vector
    y_same = y-(b/2)

    # Calculating the left and right x component of the spring force
    Fx1 = Fsp1 * (leftx / L1)
    Fx2 = Fsp2 * (rightx / L2)

    # Setting up the total x equation
    Fx = Fx1 - Fx2

    # Calculating the left and right y component of the spring force
    Fy1 = Fsp1 * (y_same / L1)
    Fy2 = Fsp2 * (y_same / L2)

    # Setting up the total x equation
    Fy = Fy1 + Fy2 - 9.81*m
    # try 2 * Fy to make Fy's equal
    Totalx = Fx1 - Fx2
    Totaly = Fy1 + Fy2 - 9.81*m
    Totalx = (Fx.subs({x:xxx, y:yyy, a:4.0,b:2.0, H:20.0, k1:100.0, k2:100.0})).evalf()
    Totaly = (Fy.subs({x:xxx, y:yyy, a:4.0,b:2.0, H:20.0, k1:100.0, k2:100.0, m:12.0})).evalf()
    abv = sympy.solve([sympy.Eq(Totalx, 0.0), sympy.Eq(Totaly, 0.0)], [Fsp1,Fsp2])
    l1 = abv[Fsp1]
    l2 = abv[Fsp2]
    return l1,l2

I_plate = me.inertia(N, 0, 0, Izz)
inertia_plate = (I_plate, G)

Plate = me.RigidBody('Plate', G, B, M, inertia_plate)

grav_force_plate = (G, -M * g * N.y)

Length1 = P1.pos_from(O1).magnitude()
Length2 = P2.pos_from(O2).magnitude()

P1_vector = P1.pos_from(O1).normalize()
P2_vector = P2.pos_from(O2).normalize()

left_force = (P1, -(F1*P1_vector))
right_force = (P2, -(F2*P2_vector))


coordinates = [x, y, beta]
speeds= [x_dot, y_dot, beta_dot]
kane = me.KanesMethod(N, coordinates, speeds, kde)

loads = [grav_force_plate, left_force, right_force]

fr, frstar = kane.kanes_equations(loads, [Plate])

Mass = kane.mass_matrix_full
f = kane.forcing_full

plate_width = 4.0
plate_height = 2.0
mass_of_plate = 10.0
rod_length = 3.0
mass_of_rod = 2.0
inertia_of_plate = (plate_width**2 + plate_height**2) * (mass_of_plate/12.0)

sys = System(kane)
sys.constants = {
                 M:12.0,
                 g:9.81,
                 F1:Lengths_and_Moments(inital_x, inital_y)[0],
                 F2:Lengths_and_Moments(inital_x, inital_y)[1],
                 H:20.0,
                 a:4.0,
                 b:2.0,
                 Izz:inertia_of_plate,
                 }
sys.initial_conditions = {x:inital_x, y:inital_y, beta:0}
sys.times = np.linspace(0.0, 20.0, 1000)

y = sys.integrate()

sim_time = np.linspace(0.0, 20.0, 1000)
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

plt.show()


