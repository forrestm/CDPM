
# coding: utf-8

# In[1]:

from sympy import symbols, init_printing
import sympy
import sympy.physics.mechanics as me
from pydy.system import System
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
init_printing(False)


# In[2]:

# Create the symbols
x, z, e, theta, phi = me.dynamicsymbols('x z e theta phi')
x_dot, z_dot, e_dot = me.dynamicsymbols('x_dot z_dot e_dot')
theta_dot, phi_dot = me.dynamicsymbols('theta_dot phi_dot')
H, a, b, M, m, g, k, t = sympy.symbols('H a b M m g k_{cable} t')
Ip, Ir, c, r, p, kr, cr, D = sympy.symbols('I_{plate} I_{rod} c r p k_{rod} c_{rod} D')
L1, L2, e_offset, k_C = sympy.symbols('L1 L2 e_{offset} k_{CFrame}')

# Create the frames Z+ points down, and Y+ is out of the screen
A = me.ReferenceFrame('A')
B = A.orientnew('B', 'Axis', [theta, A.y])
C = A.orientnew('C', 'Axis', [phi, A.y])

# Create the frames angular velocity
B.set_ang_vel(A, theta_dot * A.y)
C.set_ang_vel(A, phi_dot * A.y)

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


# In[ ]:




# In[4]:

# Set up the kinematic differential equations
kde = [x_dot - x.diff(t),
       z_dot - z.diff(t),
       e_dot - e.diff(t),
       theta_dot - theta.diff(t),
       phi_dot - phi.diff(t)]

# Create the plate inertial tensor
I_plate = me.inertia(A, 0, Ip, 0)
inertia_plate = (I_plate, G)

# Create the rod inertial tensor
I_rod = me.inertia(A, 0, Ir, 0)
inertia_rod = (I_rod, Gr)

# Create the Rigid Bodies
Plate = me.RigidBody('Plate', G, B, M, inertia_plate)
Rod = me.RigidBody('Rod', Gr, C, m, inertia_rod)


# In[5]:

# These functions do not allow the springs or dampers to push
K1 = lambda LamLenK1: k * (LamLenK1 >= L1)
K2 = lambda LamLenK2: k * (LamLenK2 >= L2)
C1 = lambda LamLenC1: c * (LamLenC1 >= L1)
C2 = lambda LamLenC2: c * (LamLenC2 >= L2)


# In[6]:

# Creating the forces acting on the body
grav_force_plate = (G, M * g * A.z)
grav_force_rod = (Gr, m * g * A.z)

spring_force_B1 = (B1, s1_vector * K1(Length1) * (L1 - Length1))
spring_force_B2 = (B2, s2_vector * K2(Length2) * (L2 - Length2))

spring_rod = (Gr, -kr * (e + e_offset) * C.z)
spring_rod_on_plate = (G, kr * (e + e_offset) * C.z)

damper_rod = (Gr, -cr * e_dot * C.z)
damper_rod_on_plate = (G, cr * e_dot * C.z)

B1_velocity = x_dot*A.x + z_dot*A.z - b*theta_dot/2*B.x + a*theta_dot/2*B.z
B2_velocity = x_dot*A.x + z_dot*A.z - b*theta_dot/2*B.x - a*theta_dot/2*B.z

B1_damping = (-C1(Length1) * B1_velocity & s1_vector) * s1_vector
B2_damping = (-C2(Length2) * B2_velocity & s2_vector) * s2_vector

damp_B1 = (B1, B1_damping)
damp_B2 = (B2, B2_damping)

B_frame_damp = (B, -5 * theta_dot * B.y)
C_frame_damp = (C, -5 * phi_dot * C.y)

C_frame_spring = (C, k_C * (theta - phi) * C.y)
C_frame_spring_on_B = (B, k_C * (phi - theta) * B.y)

loads = [grav_force_plate,
         grav_force_rod,
         spring_force_B1,
         spring_force_B2,
         spring_rod,
         spring_rod_on_plate,
         damper_rod,
         damper_rod_on_plate,
         damp_B1,
         damp_B2,
         B_frame_damp,
#          C_frame_damp,
         C_frame_spring,
         C_frame_spring_on_B]


# In[7]:

# Setting up the coordinates, speeds, and creating KanesMethod
coordinates = [x, z, e, theta, phi]
speeds = [x_dot, z_dot, e_dot, theta_dot, phi_dot]
kane = me.KanesMethod(A, coordinates, speeds, kde)

# Creating Fr and Fr_star
fr, frstar = kane.kanes_equations(loads, [Plate, Rod])

# Creating the PyDy System
sys = System(kane)


# In[8]:

# Assign the constants
workspace_width = 10.4
runtime = 30

plate_width  = 0.5
plate_height = 0.3


rod_radius   = 0.02
rod_length   = 2.91

plate_inertia = (plate_width**2 + plate_height**2) * (plate_mass/12.0)
rod_inertia = (rod_mass/12)*(3*rod_radius**2 + rod_length**2)

Length1_values = np.load('Length1_values.npy')
Length2_values = np.load('Length2_values.npy')
theta_values_heat_np = np.load('theta_values_heat_np.npy')
x_values_heat_np = np.load('x_values_heat_np.npy')
z_values_heat_np = np.load('z_values_heat_np.npy')

true_x_pre = []
true_z_pre = []
true_e_pre = []
true_theta_pre = []
true_phi_pre = []
for i in range(10000):
  sys.constants = {M: plate_mass,
                   m: rod_mass,
                   g: 9.81,
                   H: workspace_width,
                   a: plate_width,
                   b: plate_height,
                   kr: rod_spring,
                   cr: rod_damper,
                   k : cable_k,
                   c : cable_c,
                   Ip: plate_inertia,
                   Ir: rod_inertia,
                   L1: Length1_values[i],
                   L2: Length2_values[i],
             e_offset: e_offset_scalar,
                  k_C: spring_on_frame_C
                   }
  sys.initial_conditions = {x:x_values_heat_np[i],
                            z:z_values_heat_np[i],
                            theta:theta_values_heat_np[i,0],
                            e:e_equil, 
                            phi:theta_values_heat_np[i,0]}
  sys.times = np.linspace(0.0, runtime, runtime * 30)
  sys.generate_ode_function(generator='cython')
  resp = sys.integrate()
  true_x_pre.append(resp[:,0][-1])
  true_z_pre.append(resp[:,1][-1])
  true_e_pre.append(resp[:,2][-1])
  true_theta_pre.append(resp[:,3][-1])
  true_phi_pre.append(resp[:,4][-1])
  print(i)
true_x = np.asarray(true_x_pre)
true_z = np.asarray(true_z_pre)
true_e = np.asarray(true_e_pre)
true_theta = np.asarray(true_theta_pre)
true_phi = np.asarray(true_phi_pre)

np.save('true_x', true_x)
np.save('true_z', true_z)
np.save('true_e', true_e)
np.save('true_theta', true_theta)
np.save('true_phi', true_phi)
