'''
This file takes the EOMs and applies an S curve to simulate
cable motion, then plots the response. It can also apply input shaping to the
model and plot that response.

March 18:
I have updated the EOM they are corrected for the negative issue.
'''

from scipy.integrate import odeint
import sympy
import numpy as np
from numpy import sin, cos, sqrt
# import seaborn as sns
# sns.set_context("talk", font_scale=1.2)
import matplotlib.pyplot as plt
import InputShaping as shaping

# Where to store the CSVs
filepath = '/Users/forrest/Documents/CRAWLAB-Student-Code/Forrest Montgomery/CDPM/Motion_Calculations/CSV'

####### Variables to Change ##############

# Time to get to Point
Risetime = 3.0

# Time to start Moving
time_begin = 1.0

# Mass
mass = 1.0

# K of Cables
cable_K = 10.0

# C of Cables
cable_C = 1.0

# K of Rod
rod_K = 10.0

# C of Rod
rod_C = 0.1

# Length of Rod
rod_length = 3.0

# Time to plot response
endtime = 20.0
#########################################

inertia = rod_length**2 * (1.0/3.0) * mass

def eq_of_motion(w, t, p):
    """
    Defines the differential equations for a planer CDPM.

    Arguments:
        w :  vector of the state variables:
                  w = [x, y, b, x_dot, y_dot, b_dot]
        t :  time
        p :  vector of the parameters:
                  p = [m, k, g, H, c, D, t, Izz, k_beta, c_beta, L_1_init,
                       L_2_init]

    Returns:
        sysODE : An list representing the system of equations of motion
                 as 1st order ODEs
    """
    x, y, b, x_dot, y_dot, beta_dot = w
    m, k, g, H, c, D, Izz, k_beta, c_beta, Begin1, Begin2,  Amp1, Amp2, RiseTime, StartTime, Shaper = p

    # Create sysODE = (x', theta', x_dot', theta_dot'):
    sysODE = [x_dot,
              y_dot,
              beta_dot,
              (D*beta_dot**2*m*sin(b)/2 - D*m*(-D*g*m*sin(b)/2 - D*(D*beta_dot**2*m*sin(b)/2 - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*cos(b)/2 - D*(-D*beta_dot**2*m*cos(b)/2 - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*sin(b)/2 - 1.0*b*k_beta - beta_dot*c_beta - m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*(-beta_dot*x_dot*sin(b) + beta_dot*y_dot*cos(b))/2)/2 + m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*beta_dot*(-x_dot*sin(b) + y_dot*cos(b))/2)/2)*cos(b)/(2*(-D**2*m*sin(b)**2/4 - D**2*m*cos(b)**2/4 + D**2*m/4 + Izz)) - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))/m,
              (-D*beta_dot**2*m*cos(b)/2 - D*m*(-D*g*m*sin(b)/2 - D*(D*beta_dot**2*m*sin(b)/2 - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*cos(b)/2 - D*(-D*beta_dot**2*m*cos(b)/2 - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*sin(b)/2 - 1.0*b*k_beta - beta_dot*c_beta - m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*(-beta_dot*x_dot*sin(b) + beta_dot*y_dot*cos(b))/2)/2 + m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*beta_dot*(-x_dot*sin(b) + y_dot*cos(b))/2)/2)*sin(b)/(2*(-D**2*m*sin(b)**2/4 - D**2*m*cos(b)**2/4 + D**2*m/4 + Izz)) - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))/m,
              (-D*g*m*sin(b)/2 - D*(D*beta_dot**2*m*sin(b)/2 - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*cos(b)/2 - D*(-D*beta_dot**2*m*cos(b)/2 - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init(t, p) + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init(t, p) + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*sin(b)/2 - 1.0*b*k_beta - beta_dot*c_beta - m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*(-beta_dot*x_dot*sin(b) + beta_dot*y_dot*cos(b))/2)/2 + m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*beta_dot*(-x_dot*sin(b) + y_dot*cos(b))/2)/2)/(-D**2*m*sin(b)**2/4 - D**2*m*cos(b)**2/4 + D**2*m/4 + Izz)
              ]
    return sysODE

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
    coord = coord[:n]
    return coord

'''
This is to map out the entire workspace instead of just a circle
'''
X_and_Y_points = 20

X = np.linspace(1.0, 19.0, num=X_and_Y_points)
Y = np.linspace(1.0, 14.0, num=X_and_Y_points)
n = X_and_Y_points
dims=np.array((0,0))
for i in range(n):
    x_i = np.repeat(X[i],n)
    total = np.column_stack((x_i,Y))
    dims = np.vstack((dims,total))
dims = dims[1:]

############## Variables to Change #########################
# Either comment out the circle or the dims to start getting points

radius_of_circle = 4

number_of_points = 20

# points = circle(radius_of_circle, number_of_points)

points = dims

############################################################

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

def L_1_init(t, p):
    """
    Defines the Length 1 input to the system.

    Arguments:
        t : current time step
        p : vector of parameters
    """
    m, k, g, H, c, D, Izz, k_beta, c_beta, Begin1, Begin2, Amp1, Amp2, RiseTime, StartTime, Shaper = p

    if Shaper == []:
        L_1_init = s_curve(t, Begin1, Amp1, RiseTime, StartTime)
    else:
        L_1_init = shaping.shaped_input(s_curve, t, Shaper, Begin1, Amp1, RiseTime,
                                        StartTime)
    return L_1_init


def L_2_init(t, p):
    """
    Defines the Length 2 input to the system.

    Arguments:
        t : current time step
        p : vector of parameters
    """
    m, k, g, H, c, D, Izz, k_beta, c_beta, Begin1, Begin2, Amp1, Amp2, RiseTime, StartTime, Shaper = p

    if Shaper == []:
        L_2_init = s_curve(t, Begin2, Amp2, RiseTime, StartTime)
    else:
        L_2_init = shaping.shaped_input(s_curve, t, Shaper, Begin2, Amp2, RiseTime,
                                        StartTime)
    return L_2_init


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
      Begin : The beginning value

    Returns :
      The command at the current timestep or an array representing the command
      over the times given (if CurrTime was an array)
    """

    Amp = Amp - Begin

    scurve = 2.0 * ((CurrTime - StartTime)/RiseTime)**2 * (CurrTime-StartTime >=
             0) * (CurrTime-StartTime < RiseTime/2) + (-2.0 * ((CurrTime -
             StartTime)/RiseTime)**2 + 4.0 * ((CurrTime - StartTime) /RiseTime)
             - 1.0) * (CurrTime-StartTime >= RiseTime/2) * (CurrTime-StartTime <
             RiseTime) + 1.0 * (CurrTime-StartTime >= RiseTime)

    return (Amp * scurve) + Begin

################## Various Shapers ######################################

# Going with 500rpm
# example_shaper = shaping.EI(0.329, 0.101)
# example_shaper = shaping.EI(0.4323345411805539, 0.25)
example_shaper = shaping.ZV_EI_2mode(0.25946057132, 0.00869237035333, 0.47558317356, 0.203271432775)
# example_shaper = shaping.ZVD(0.32, 0.032272678421545214)
# example_shaper = shaping.ZV_EI_2mode(.466053, 0.032272, .466053, 0.032272)
# example_shaper = shaping.UMZVD(.46605358040201028, 0.032272678421545214)
# example_shaper = shaping.ZVD_2mode(0.40160804, 0.07159188, 0.047746, 0.0)
# Shaper = example_shaper.shaper
#########################################################################


################################ UNSHAPED ##################################
n = np.shape(points)[0]

for i in range(n):

    # ODE solver parameters
    abserr = 1.0e-9
    relerr = 1.0e-9
    max_step = 0.001
    stoptime = endtime
    numpoints = 1000

    # Create the time samples for the output of the ODE solver
    t = np.linspace(0.0, stoptime, numpoints)

    # Set up simulation parameters
    m = mass
    k = cable_K
    g = 9.81
    H = 20.0
    c = cable_C
    D = rod_length
    Izz = inertia
    k_beta = rod_K
    c_beta = rod_C

    L1y, L2y = True_length(points[0:,0][i],points[0:,1][i])
    x_init = points[0:,0][i]                       # initial position
    x_dot_init = 0.0                    # initial velocity
    y_init = points[0:,1][i]                   # initial angle
    y_dot_init = 0.0                # initial angular velocity
    beta_init = 0.0                        # initial position
    beta_dot_init = 0.0

    # Set up the parameters for the input function
    Begin1 = L1y               # Initial Length 1 of Cable (m)
    Begin2 = L2y              # Initial Length 2 of Cable (m)
    Amp1 = 11.6825                  # Final Cable 1 Length (m)
    Amp2 = 11.6825                 # Final Cable 2 Length(m)
    StartTime = time_begin              # Time the cables will start moving (s)
    RiseTime = Risetime               # Time for to move to takes (s)
    Shaper = []

    # Pack the parameters and initial conditions into arrays
    p = [m, k, g, H, c, D, Izz, k_beta, c_beta, Begin1, Begin2, Amp1, Amp2,
         RiseTime, StartTime, Shaper]

    x0 = [x_init, y_init, beta_init, x_dot_init, y_dot_init, beta_dot_init]

    resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr, hmax=max_step)
    np.savetxt(filepath+"response"+str(i)+".csv", resp, delimiter=",")
    # np.savetxt("response"+str(i)+".csv", resp, delimiter=",")
    print(i)


################################ UNSHAPED ##################################
n = np.shape(points)[0]

for i in range(n):

    L1y, L2y = True_length(points[0:,0][i],points[0:,1][i])
    x_init = points[0:,0][i]                       # initial position
    x_dot_init = 0.0                    # initial velocity
    y_init = points[0:,1][i]                   # initial angle
    y_dot_init = 0.0                # initial angular velocity
    beta_init = 0.0                        # initial position
    beta_dot_init = 0.0

    # Set up the parameters for the input function
    Begin1 = L1y                # Initial Length 1 of Cable (m)
    Begin2 = L2y               # Initial Length 2 of Cable (m)
    Amp1 = 11.6825                 # Final Cable 1 Length (m)
    Amp2 = 11.6825                 # Final Cable 2 Length(m)
    StartTime = time_begin              # Time the cables will start moving (s)
    RiseTime = Risetime               # Time for to move to takes (s)

    # Pack the parameters and initial conditions into arrays
    p = [m, k, g, H, c, D, Izz, k_beta, c_beta, Begin1, Begin2, Amp1, Amp2,
         RiseTime, StartTime, example_shaper.shaper]

    x0 = [x_init, y_init, beta_init, x_dot_init, y_dot_init, beta_dot_init]

    resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr, hmax=max_step)
    np.savetxt(filepath+"response_s"+str(i)+".csv", resp, delimiter=",")
    print(i)
#############################################################################
#
# n = np.shape(points)[0]
# amps = np.array([(0,0)])
# percent = np.array([0])
# for i in range(n):
#     resp = np.genfromtxt('response'+str(i)+'.csv', delimiter=',')
#     shaped = np.genfromtxt('response_s'+str(i)+'.csv', delimiter=',')
#     itemindex = np.where(shaped[:,0] >= 10)
#     shaped_ends = itemindex[:1][0][0]
#
#     itemindex_resp = np.where(resp[:,0] >= 10)
#     unshaped_ends = itemindex[:1][0][0]
#
#     beta_max = np.abs(np.max(resp[:,2][unshaped_ends:]))
#     beta_min = np.abs(np.min(resp[:,2][unshaped_ends:]))
#     beta_amp = np.abs(beta_max - beta_min)
#
#     beta_shaped_max = np.abs(np.max(shaped[:,2][shaped_ends:]))
#     beta_shaped_min = np.abs(np.min(shaped[:,2][shaped_ends:]))
#     beta_shaped_amp = np.abs(beta_shaped_max - beta_shaped_min)
#
#     percent_vib = (beta_shaped_amp / beta_amp) * 100
#     amps = np.append(amps, [(beta_amp, beta_shaped_amp)], axis=0)
#     percent = np.append(percent, [percent_vib])
# amps = amps[1:,:]
#
# # np.savetxt("percent.csv", percent, delimiter=",")
# # np.savetxt("amps.csv", amps, delimiter=",")
#
# plt.figure(0)
# plt.plot(percent)
# plt.title('Percent Vibration')
# plt.xlabel('Move Number')
# plt.ylabel('Percent Vibration')
# plt.show()
#
# plt.figure(1)
# plt.plot(amps[:,0], label='Unshaped')
# plt.plot(amps[:,1], label='Shaped')
# plt.title('Total Amplitude after Command')
# plt.legend()
# plt.xlabel('Move Number')
# plt.ylabel('Amplitude')
# plt.show()

# plt.figure(3)
# # plt.subplot(212)
# plt.plot(resp[:,0], resp[:,1])
# plt.plot(resp[:,0] + 0.5/2 * np.sin(resp[:,2]) , resp[:,1] + 0.5/2 *
#                                     np.cos(resp[:,2]), label='Unshaped')
# plt.plot(resp[:,0] + 1/2 * np.sin(resp[:,2]) , resp[:,1] + 1/2 *
#                                   np.cos(resp[:,2]), label='Unshaped')
# plt.legend()
# plt.title('Front View Motion')
# # plt.ylim(25,0)
# plt.ylim(25,0)
# # plt.xlim(0,20)
# plt.xlim(0,20)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# plt.subplot(2, 1, 1)
# plt.plot(t, s_curve(t, Begin1, Amp1, RiseTime, StartTime))
# plt.plot(t, shaping.shaped_input(s_curve, t, Shaper, Begin1, Amp1, RiseTime, StartTime))
# plt.title('Length 1 Displacement')
# plt.ylabel('Length (s)')
# plt.ylabel('Time (s)')
#
# plt.subplot(2, 1, 2)
# plt.plot(t, s_curve(t, Begin2, Amp2, RiseTime, StartTime))
# plt.plot(t, shaping.shaped_input(s_curve, t, Shaper, Begin2, Amp2, RiseTime, StartTime))
# plt.title('Length 2 Displacement')
# plt.ylabel('Length (s)')
# plt.ylabel('Time (s)')
# plt.show()
