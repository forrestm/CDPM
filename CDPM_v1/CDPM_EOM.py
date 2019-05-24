"""
This program will create the EOM for a CDPM with a solid rod pendulum attached
to the end-effector. It will then export the equations to a csv file.
"""
import sympy
from sympy import symbols, init_printing
import sympy.physics.mechanics as me
init_printing(use_latex='mathjax')

# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt
import numpy as np

# Create the variables
x, y, beta = me.dynamicsymbols('x, y, beta')

# Create the velocities
x_dot, y_dot, beta_dot = me.dynamicsymbols('x, y, beta', 1)

# Create the constants
m, k, L, g, H, c, D, t = sympy.symbols('m k L g H c D t')
Izz, k_beta, c_beta = sympy.symbols('Izz k_beta c_beta')
L_1_init, L_2_init = sympy.symbols('L_1_init L_2_init')
'''
m = mass
k = spring k
L = spring equilibrium length
g = gravity
c = spring c
c_beta = rotational c
k_beta = rotational k
D = rod length
Izz = moment of Inertia about the end of a rod
'''

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
# O2 = O1.locatenew('O_2', H * N.x)
O2 = me.Point('O_2')
O2.set_pos(O1, H * N.x)
O2.set_vel(N, 0)

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

# Create the height from the center of gravity to the datum
h = G.pos_from(O1) & N.y

# The forces at the connection point
forceP = c * (x_dot + y_dot) * L1_vector() + c * (-x_dot + y_dot) * L2_vector()

# The forces on the beta frame
forceB = c_beta * beta_dot * N.z

rod.potential_energy = (-m * g * h + 0.5 * k * (L1() - L_1_init)**2 + 0.5 *
                        k *(L2() - L_2_init)**2 + 0.5 * k_beta * beta**2)

Lag = me.Lagrangian(N, rod)

LM = me.LagrangesMethod(Lag, [x, y, beta], forcelist=[(P, forceP), (B, forceB)], frame=N)


EqMotion = LM.form_lagranges_equations()

lrhs = LM.rhs()
EOM = lrhs.subs({x:'x',y:'y', beta:'b', x_dot:'x_dot',
                          y_dot:'y_dot',beta_dot:'beta_dot'})
print(EOM)

np.savetxt("EOM.csv", EOM, delimiter=",", fmt="%s")
