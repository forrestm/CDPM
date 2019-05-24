'''
This file will compute the EOM for several points then solve them with and
without a Shaper and export each response into a CSV
'''

from sympy import symbols, init_printing
import sympy
import sympy.physics.mechanics as me
init_printing(use_latex='mathjax')
import seaborn as sns
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt
from scipy.integrate import ode, odeint
import numpy as np
import InputShaping as shaping

# Where to store the CSVs
filepath = '/Users/forrest/Desktop/CDPM/Motion_Calculations/CSV/'

# Create the variables
x, y, beta = me.dynamicsymbols('x, y, beta')

# Create the velocities
x_dot, y_dot, beta_dot = me.dynamicsymbols('x, y, beta', 1)

# Create the constants
m, k, L, g, H, c, D, t = sympy.symbols('m k L g H c D t')
Izz, k_beta, c_beta = sympy.symbols('Izz k_beta c_beta')
L_1_init, L_2_init = sympy.symbols('L_1_init L_2_init')
Le1, Le2 = sympy.symbols('Le1 Le2')
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

# Create the height from the center of gravity to the datum
h = G.pos_from(O1) & N.y

# The forces at the connection point
forceP = c * (x_dot + y_dot) * L1_vector() + c * (-x_dot + y_dot) * L2_vector()

# The forces on the beta frame
forceB = -c_beta * beta_dot * N.z

rod.potential_energy = (-m * g * h + 0.5 * k * (L1() - L_1_init)**2 + 0.5 *
                        k *(L2() - L_2_init)**2 + 0.5 * k_beta * beta**2)

Lag = me.Lagrangian(N, rod)

LM = me.LagrangesMethod(Lag, [x, y, beta], forcelist=[(P, forceP), (B, forceB)],
                        frame=N)


EqMotion = LM.form_lagranges_equations()

lrhs = LM.rhs()
# print(lrhs)


############### Circle Function ##############################
def circle(radius, count):
    '''
    This will take a radius and an amount of points, and create coordinates around
    the center point (10, 7.5). So the radius cannot be larger than 7.5.
    '''

    coord = np.array([(10 + radius, 7.5)])
    origin = np.array([(10, 7.5)])
    circumfrence = 2 * np.pi * radius
    amount = circumfrence / count
    n = count
    theta = amount / radius
    for i in range(n):
        y = 7.5 - radius * np.sin(theta*i)
        x = 10 + radius * np.cos(theta*i)
        coord = np.append([(x,y)], coord, axis=0)
    coord = np.reshape(coord, [count+1, 2])
    return coord

############## Variables to Change #########################

radius_of_circle = 4

number_of_points = 80

points = circle(radius_of_circle, number_of_points)

############################################################

############### Length Function ####################################
def True_length(x,y):
    '''
    This calculates the True lengths of the cables, for any given X and Y the geometric cable
    lengths will not be correct because of the spring stretch. This calculates the cable length
    that will stretch to the X and Y position.

    Y has to be above 1 or the cables cannot reach that position.

    The sympy solver is very slow it would be more efficent to use numpy,
    however this only has to be run once to get the lengths so the convience of this slow
    algebric solver is ok.
    '''
    k=10
    h=20
    m=1.0
    Fy = np.array([[k*y/np.sqrt(x**2 + y**2), k*y/np.sqrt(y**2 + (h - x)**2)],
                   [k*x/np.sqrt(x**2 + y**2), -h*k/np.sqrt(y**2 + (h - x)**2
                    )+k*x/np.sqrt(y**2 + (h - x)**2)]])
    a = np.array([2*k*y - 9.81*m,-h*k + 2*k*x])
    x = np.linalg.solve(Fy,a)
    return x
###############################################################################

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
    scurve = 2.0 * ((CurrTime - StartTime)/RiseTime)**2 * (CurrTime-StartTime >= 0) * (CurrTime-StartTime < RiseTime/2)             +(-2.0 * ((CurrTime - StartTime)/RiseTime)**2 + 4.0 * ((CurrTime - StartTime)/RiseTime) - 1.0) * (CurrTime-StartTime >= RiseTime/2) * (CurrTime-StartTime < RiseTime)             + 1.0 * (CurrTime-StartTime >= RiseTime)

    return (Amp * scurve) + Begin


# Various Shapers

# Going with 500rpm
# example_shaper = shaping.EI(0.329, 0.101)
example_shaper = shaping.EI(0.47191544453, 0.1)
# example_shaper = shaping.ZVD(0.32, 0.032272678421545214)
# example_shaper = shaping.ZV_EI_2mode(.466053, 0.032272, .466053, 0.032272)
# example_shaper = shaping.UMZVD(.46605358040201028, 0.032272678421545214)
# example_shaper = shaping.ZVD_2mode(0.40160804, 0.07159188, 0.047746, 0.0)
Shaper = example_shaper.shaper

# a = shaping.shaped_input(s_curve, t, Shaper, 10.0,13.0,5.0,3.0)


length_of_rod = 3.0
mass = 1.0
inertia = length_of_rod**2 * (1.0/3.0) * mass

############### UNSHAPED #####################################################
n = np.shape(points)[0]

for i in range(n):

  L1y, L2y = True_length(points[:,0][i],points[:,1][i])


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
                k : 10.0,
                # L_1_init: shaping.shaped_input(s_curve, t, Shaper, L1y,11.6825,9.0,3.0),
                # L_2_init: shaping.shaped_input(s_curve, t, Shaper, L2y,11.6825,9.0,3.0),
                L_1_init:s_curve(t, L1y, 11.6825, 9.0, 3.0),
                L_2_init:s_curve(t, L2y, 11.6825, 9.0, 3.0),
                H : 20.0,
                c : 1.0,
                D : length_of_rod,
                Izz: inertia,
                k_beta: 1.0,
                c_beta: 0.1}

  # set this parameter to enable array output from sympy.lambdify
  mat2array = [{'ImmutableMatrix': np.array}, 'numpy']

  # Create a function from the equations of motion
  # Here, we substitude the states and parameters as appropriate prior to the lamdification
  eq_of_motion = sympy.lambdify((t, w),
                                lrhs.subs(sub_params),
                                modules = mat2array)


  # In[32]:

  end_time = 20.0

  x0 = [points[:,0][i],points[:,1][i], 0.0, 0.0, 0.0, 0.0]

  # Positive beta is to the left when facing the structure
  sim_time = np.linspace(0.0, end_time, 1000)


  # In[33]:

  # Set up the initial point for the ode solver
  r = ode(eq_of_motion).set_initial_value(x0, sim_time[0])

  # define the sample time
  dt = sim_time[1] - sim_time[0]

  # pre-populate the response array with zeros
  response_i = np.zeros((len(sim_time), len(x0)))

  # Set the initial index to 0
  index = 0

  # Now, numerically integrate the ODE while:
  #   1. the last step was successful
  #   2. the current time is less than the desired simluation end time
  while r.successful() and r.t < sim_time[-1]:
      response_i[index, :] = r.y
      r.integrate(r.t + dt)
      index += 1
  print(i)
  # np.savetxt("response_s"+str(i)+".csv", response_i, delimiter=",")
  np.savetxt(filepath+"response"+str(i)+".csv", response_i, delimiter=",")
  # np.savetxt("sim_time.csv", sim_time, delimiter=",")

  #############################################################################


############### SHAPED #####################################################
n = np.shape(points)[0]

for i in range(n):

  L1y, L2y = True_length(points[:,0][i],points[:,1][i])


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
                k : 10.0,
                L_1_init: shaping.shaped_input(s_curve, t, Shaper, L1y,11.6825,9.0,3.0),
                L_2_init: shaping.shaped_input(s_curve, t, Shaper, L2y,11.6825,9.0,3.0),
                # L_1_init:s_curve(t, L1y, 11.6825, 9.0, 3.0),
                # L_2_init:s_curve(t, L2y, 11.6825, 9.0, 3.0),
                H : 20.0,
                c : 1.0,
                D : length_of_rod,
                Izz: inertia,
                k_beta: 1.0,
                c_beta: 0.1}

  # set this parameter to enable array output from sympy.lambdify
  mat2array = [{'ImmutableMatrix': np.array}, 'numpy']

  # Create a function from the equations of motion
  # Here, we substitude the states and parameters as appropriate prior to the lamdification
  eq_of_motion = sympy.lambdify((t, w),
                                lrhs.subs(sub_params),
                                modules = mat2array)


  # In[32]:

  end_time = 20.0

  x0 = [points[:,0][i],points[:,1][i], 0.0, 0.0, 0.0, 0.0]

  # Positive beta is to the left when facing the structure
  sim_time = np.linspace(0.0, end_time, 1000)


  # In[33]:

  # Set up the initial point for the ode solver
  r = ode(eq_of_motion).set_initial_value(x0, sim_time[0])

  # define the sample time
  dt = sim_time[1] - sim_time[0]

  # pre-populate the response array with zeros
  response_s_i = np.zeros((len(sim_time), len(x0)))

  # Set the initial index to 0
  index = 0

  # Now, numerically integrate the ODE while:
  #   1. the last step was successful
  #   2. the current time is less than the desired simluation end time
  while r.successful() and r.t < sim_time[-1]:
      response_s_i[index, :] = r.y
      r.integrate(r.t + dt)
      index += 1
  print(i)

  np.savetxt(filepath+"response_s"+str(i)+".csv", response_s_i, delimiter=",")
  # np.savetxt("response"+str(i)+".csv", response_i, delimiter=",")
  # np.savetxt("sim_time.csv", sim_time, delimiter=",")
