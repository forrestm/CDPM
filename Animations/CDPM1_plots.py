'''
This file will compute the EsOM then solve with and without a Shaper, and then
animate the motion and export the video
'''

from sympy import symbols, init_printing
import sympy
import sympy.physics.mechanics as me
# import seaborn as sns
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
import numpy as np
import InputShaping as shaping

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

####### Variables to Change ##############

# Beginning Point
# X_begin = 5.32
# Y_begin = 7.5

X_begin = 3
Y_begin = 4

# Ending Point
X_end = 10
Y_end = 7.5

# Time to get to Point was 3/4
Risetime = np.sqrt((X_begin-10)**2 + (Y_begin - 7.5)**2)*(0.5)

# Time to start Moving
time_begin = 1.0

# Mass
mass = 10.0

# K of Cables
cable_K = 100.0

# C of Cables
cable_C = 10.0

# K of Rod
rod_K = 1.0

# C of Rod
rod_C = 1.0

# Length of Rod
rod_length = 3.0

# Time to plot response
endtime = 50.0
#########################################

inertia = rod_length**2 * (1.0/3.0) * mass

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

L1_begin, L2_begin = Lengths(X_begin, Y_begin)
L1_end, L2_end = Lengths(X_end, Y_end)

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

#########################################
# f_c = sympy.Matrix([(D*sympy.sin(beta)+x)**2 + (D*sympy.cos(beta)+y)**2 - D**2])
#########################################

rod.potential_energy = (-m * g * h + 0.5 * k * (L1() - L_1_init)**2 + 0.5 *
                        k *(L2() - L_2_init)**2 + 0.5 * k_beta * beta**2)

Lag = me.Lagrangian(N, rod)

LM = me.LagrangesMethod(Lag, [x, y, beta], forcelist=[(P, forceP), (B, forceB)], frame=N)

#########################################
# LM = me.LagrangesMethod(Lag, [x, y, beta], hol_coneqs=f_c, forcelist=[(P, forceP),
#                         (B, forceB)], frame=N)
#########################################

EqMotion = LM.form_lagranges_equations()

lrhs = LM.rhs()
# print(lrhs)

def s_curve(CurrTime, Begin, Amp, RiseTime, StartTime):
    """
    This was copied from Dr. Vaughan's Input shaping Library
    I edited it to allow for a beginning value.

    Function to generate an s-curve command

    Arguments:
      CurrTime : The current timestep or an array of times
      Amp : The magnitude of the s-curve (or final setpoint)
      RiseTime : The rise time of the curve
      StartTime : The time that the command should StartTime
      Begin : The beginnning value

    Returns :
      The command at the current timestep or an array representing the command
      over the times given (if CurrTime was an array)
    """

    Amp = Amp - Begin
    scurve = 2.0 * ((CurrTime - StartTime)/RiseTime)**2 * (CurrTime-StartTime >= 0) * (CurrTime-StartTime < RiseTime/2)+(-2.0 * ((CurrTime - StartTime)/RiseTime)**2 + 4.0 * ((CurrTime - StartTime)/RiseTime) - 1.0) * (CurrTime-StartTime >= RiseTime/2) * (CurrTime-StartTime < RiseTime)    + 1.0 * (CurrTime-StartTime >= RiseTime)

    return (Amp * scurve) + Begin


# Various Shapers

# Going with 500rpm
# example_shaper = shaping.EI(0.329, 0.101)
# These values were averaged across the entire workspace from the eigenvalues
# example_shaper = shaping.ZV_EI_2mode(0.25946057132, 0.00869237035333, 0.47558317356, 0.203271432775)
# example_shaper = shaping.ZV_EI_EI_3mode(0.2593206987035489, 0.017766920997407218,
#                                   0.4756421924129393, 0.20371676657418006,
#                                   0.7802304334889989, 0.251892133943285)
# example_shaper = shaping.EI_2mode(0.4756421924129393, 0.20371676657418006,
#                                   0.2593206987035489, 0.017766920997407218)
# example_shaper = shaping.ZVD(0.32, 0.032272678421545214)
example_shaper = shaping.ZV_EI_2mode(0.2526851, 0.01680748, 0.47542116, 0.20396849)

# example_shaper = shaping.ZV_EI_2mode(.466053, 0.032272, .466053, 0.032272)
# example_shaper = shaping.UMZVD(.46605358040201028, 0.032272678421545214)
# example_shaper = shaping.ZVD_2mode(0.40160804, 0.07159188, 0.047746, 0.0)
Shaper = example_shaper.shaper

# a = shaping.shaped_input(s_curve, t, Shaper, 10.0,13.0,5.0,3.0)


############### UNSHAPED #####################################################

# Define the states and state vector
w1, w2, w3, w4, w5, w6 = sympy.symbols('w1 w2 w3 w4 w5 w6', cls=sympy.Function)
w = [w1(t), w2(t), w3(t), w4(t), w5(t), w6(t)]

# Set up the state definitions and parameter substitution
sub_params = {x : w1(t),
              y : w2(t),
              beta: w3(t),
              x_dot: w4(t),
              y_dot: w5(t),
              beta_dot: w6(t),
              m : mass,
              g : 9.81,
              k : cable_K,
              L_1_init:s_curve(t, L1_begin, L1_end, Risetime, time_begin),
              L_2_init:s_curve(t, L2_begin, L2_end, Risetime, time_begin),
              H : 20.0,
              c : cable_C,
              D : rod_length,
              Izz: inertia,
              k_beta: rod_K,
              c_beta: rod_C}

# set this parameter to enable array output from sympy.lambdify
mat2array = [{'ImmutableMatrix': np.array}, 'numpy']

# Create a function from the equations of motion
# Here, we substitude the states and parameters as appropriate prior to the lamdification
eq_of_motion = sympy.lambdify((t, w),
                              lrhs.subs(sub_params),
                              modules = mat2array)

x0 = [X_begin, Y_begin, 0.0, 0.0, 0.0, 0.0]

# Positive beta is to the left when facing the structure
sim_time = np.linspace(0.0, endtime, 1000)


# In[33]:

# Set up the initial point for the ode solver
r = ode(eq_of_motion).set_initial_value(x0, sim_time[0])

# define the sample time
dt = sim_time[1] - sim_time[0]

# pre-populate the response array with zeros
response = np.zeros((len(sim_time), len(x0)))

# Set the initial index to 0
index = 0

# Now, numerically integrate the ODE while:
#   1. the last step was successful
#   2. the current time is less than the desired simluation end time
while r.successful() and r.t < sim_time[-1]:
    response[index, :] = r.y
    r.integrate(r.t + dt)
    index += 1

# np.savetxt("response.csv", response, delimiter=",")
# np.savetxt("response_shaped.csv", response2, delimiter=",")
# np.savetxt("sim_time.csv", sim_time, delimiter=",")

#############################################################################

############### SHAPED ######################################################

# Define the states and state vector
w1, w2, w3, w4, w5, w6 = sympy.symbols('w1 w2 w3 w4 w5 w6', cls=sympy.Function)
w = [w1(t), w2(t), w3(t), w4(t), w5(t), w6(t)]

# Set up the state definitions and parameter substitution
sub_params = {x : w1(t),
              y : w2(t),
              beta: w3(t),
              x_dot: w4(t),
              y_dot: w5(t),
              beta_dot: w6(t),
              m : mass,
              g : 9.81,
              k : cable_K,
              L_1_init: shaping.shaped_input(s_curve, t, Shaper, L1_begin,L1_end,Risetime,time_begin),
              L_2_init: shaping.shaped_input(s_curve, t, Shaper, L2_begin,L2_end,Risetime,time_begin),
              H : 20.0,
              c : cable_C,
              D : rod_length,
              Izz: inertia,
              k_beta: rod_K,
              c_beta: rod_C}

# set this parameter to enable array output from sympy.lambdify
mat2array = [{'ImmutableMatrix': np.array}, 'numpy']

# Create a function from the equations of motion
# Here, we substitude the states and parameters as appropriate prior to the lamdification
eq_of_motion = sympy.lambdify((t, w),
                              lrhs.subs(sub_params),
                              modules = mat2array)



x0 = [X_begin, Y_begin, 0.0, 0.0, 0.0, 0.0]

# Positive beta is to the left when facing the structure
sim_time = np.linspace(0.0, endtime, 1000)


# In[33]:

# Set up the initial point for the ode solver
r = ode(eq_of_motion).set_initial_value(x0, sim_time[0])

# define the sample time
dt = sim_time[1] - sim_time[0]

# pre-populate the response array with zeros
response2 = np.zeros((len(sim_time), len(x0)))

# Set the initial index to 0
index = 0

# Now, numerically integrate the ODE while:
#   1. the last step was successful
#   2. the current time is less than the desired simluation end time
while r.successful() and r.t < sim_time[-1]:
    response2[index, :] = r.y
    r.integrate(r.t + dt)
    index += 1
# In[34]:

#############################################################################


# sns.set_palette("Paired")
# sns.set_context("talk", font_scale=1.2)
# sns.set_style("darkgrid")

# Set the plot size - 3x2 aspect ratio is best
# fig = plt.figure(figsize=(6,4))
# ax = plt.gca()
# plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units font
# plt.setp(ax.get_ymajorticklabels(),fontsize=18)
# plt.setp(ax.get_xmajorticklabels(),fontsize=18)

# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')

# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
# ax.grid(True,linestyle=':', color='0.75')
# ax.set_axisbelow(True)

# Define the X and Y axis labels
# plt.xlabel('Time (s)', fontsize=22, weight='bold', labelpad=5)
# plt.ylabel('Angle (deg)', fontsize=22, weight='bold', labelpad=10)
 
# plt.plot(sim_time, np.degrees(response[:,2]), label='Unshaped', linewidth=2, linestyle='-.')
# plt.plot(sim_time, np.degrees(response2[:,2]), label ='Shaped', linewidth=2, linestyle='-')

# uncomment below and set limits if needed
# plt.xlim(0,60)
# plt.ylim(-20,20)

# Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 3, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext,fontsize=18)

# Adjust the page layout filling the page using the new tight_layout command
# plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('.pdf')

# show the figure
# plt.show()

# plt.figure(0)
# # plt.subplot(211)
# plt.plot(sim_time, response[:,1], label='Unshaped')
# plt.plot(sim_time, response2[:,1], label ='Shaped')
# plt.ylim(25,0)
# plt.legend()
# plt.title('Y Motion')
# plt.xlabel('time (s)')
# plt.ylabel('meters')
# plt.show()

# plt.figure(1)
# # plt.subplot(211)
# plt.plot(sim_time, response[:,0], label='Unshaped')
# plt.plot(sim_time, response2[:,0], label ='Shaped')
# plt.legend()
# plt.title('X Motion')
# plt.xlabel('time (s)')
# plt.ylabel('meters')
# plt.show()

# plt.figure(2)
# # plt.subplot(211)
# plt.plot(sim_time, np.degrees(response[:,2]), label='Unshaped')
# plt.plot(sim_time, np.degrees(response2[:,2]), label ='Shaped')
# plt.legend()
# # plt.plot(sim_time, np.degrees(response2[:,2]))
# plt.ylim(-20,20)
# plt.xlim(0,60)
# plt.title('Beta Motion')
# plt.xlabel('time (s)')
# plt.ylabel('Degrees')
# # plt.savefig("Beta_Motion.pdf")
# plt.show()

# plt.figure(3)
# plt.subplot(212)

x_un = response[:,0]
y_un = response[:,1]

xend_un = response[:,0] + rod_length * np.sin(response[:,2])
yend_un = response[:,1] + rod_length * np.cos(response[:,2])

R = np.sqrt((10 - xend_un)**2 + (10.5 - yend_un)**2)

# plt.figure(0)
# plt.plot(sim_time,R)
# plt.xlabel('Time (s)')
# plt.ylabel('R Displacement (m)')
# plt.xlim(0,20)
# plt.ylim(0,9)
# plt.savefig("Unshaped.pdf")
# plt.show()


# plt.plot(x_un, y_un)
# plt.plot(xp_un, yp_un, label='Unshaped')
# plt.plot(xend_un, yend_un, label='Unshaped')

x_s = response2[:,0]
y_s = response2[:,1]

xend_s = response2[:,0] + rod_length * np.sin(response2[:,2])
yend_s = response2[:,1] + rod_length * np.cos(response2[:,2])

Rs = np.sqrt((10 - xend_s)**2 + (10.5 - yend_s)**2)

# plt.figure(0)
# plt.plot(sim_time, Rs)
# plt.xlabel('Time (s)')
# plt.ylabel('R Displacement (m)')
# plt.xlim(0,20)
# plt.ylim(0,9)
# plt.savefig("Shaped.pdf")
# plt.show()

plt.figure(1)
plt.figure(0)
plt.plot(sim_time, Rs, label="Shaped")
plt.plot(sim_time, R, label="Unshaped", linestyle="--")

plt.xlabel('Time (s)')
plt.ylabel('R Displacement (m)')
plt.xlim(0,20)
plt.ylim(0,9)
plt.legend()
plt.savefig("Combined.pdf")
plt.show()
# plt.plot(x_s, y_s)
# plt.plot(xp_s, yp_s, label='Shaped')
# plt.plot(xend_s, yend_s, label='Shaped')

