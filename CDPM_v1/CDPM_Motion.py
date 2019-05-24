'''
This file takes the EOMs and applies an S curve to simulate
cable motion, it will also apply input shaping to the model and plot the two
responses.
'''

from scipy.integrate import odeint
import numpy as np
from numpy import sin, cos, sqrt
# import seaborn as sns
# sns.set_context("talk", font_scale=1.2)
import matplotlib.pyplot as plt
import InputShaping as shaping


####### Variables to Change ##############

# Beginning Point
# X_begin = 15.0
# Y_begin = 7.5

X_begin = 5.32
Y_begin = 7.5
# 7.84210526

# Ending Point
X_end = 10
Y_end = 7.5

# Time to get to Point
# To keep the velocity the same across the workspace
Risetime = np.sqrt((X_begin-10)**2 + (Y_begin - 7.5)**2)*(0.5)

# Time to start Moving
time_begin = 1.0

# Mass
mass = 15.0

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
endtime = 20.0
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



def eq_of_motion(w, t, p):
    """
    Defines the differential equations for a planar CDPM.

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
# example_shaper = shaping.ZV_EI_EI_3mode(0.2593206987035489, 0.017766920997407218,
#                                   0.4756421924129393, 0.20371676657418006,
#                                   0.7802304334889989, 0.251892133943285)
# example_shaper = shaping.ZV_EI_EI_3mode(0.26, 0.01776,
#                                   0.48, 0.2037,
#                                   0.78, 0.2518)
example_shaper = shaping.ZV_EI_2mode(0.2526851, 0.01680748, 0.47542116, 0.20396849)
# example_shaper = shaping.ZVD(0.25946057132, 0.00869237035333)
# example_shaper = shaping.ZV_EI_2mode(0.2593206987035489, 0.017766920997407218, 0.47558317356, 0.203271432775)
# example_shaper = shaping.UMZVD(0.25946057132, 0.00869237035333, 0.47558317356, 0.203271432775)
# example_shaper = shaping.ZVD_2mode(0.25946057132, 0.00869237035333, 0.47558317356, 0.203271432775)
# example_shaper = shaping.EI(0.2593206987035489, 0.017766920997407218)
# example_shaper = shaping.ZV_EI_2mode(0.25886185, 0.00929304, 0.47590239, 0.03270558)
#########################################################################


# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
numpoints = 1000

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0, endtime, numpoints)

# Set up simulation parameters
# m = 1.0
# k = 10.0
g = 9.81
H = 20.0
# c = 1.0
# D = 1.0
# Izz = 1
# k_beta = 1.0
# c_beta = 0.01
# L_1_init = s_curve(t, 15.62, 25.0, 4.0, 3.0)
# L_2_init = s_curve(t, 15.62, 20.61, 4.0, 3.0)


# Initial conditions
# x_init = 10.0                        # initial position
x_dot_init = 0.0                    # initial velocity
# y_init = 12.0                   # initial angle
y_dot_init = 0.0                # initial angular velocity
beta_init = 0.0                        # initial position
beta_dot_init = 0.0

# Set up the parameters for the input function
# Begin1 = 15.62               # Initial Length 1 of Cable (m)
# Begin2 = 15.62               # Initial Length 2 of Cable (m)
# Amp1 = 25.0                  # Final Cable 1 Length (m)
# Amp2 = 20.61                 # Final Cable 2 Length(m)
# StartTime = 3.0              # Time the cables will start moving (s)
# RiseTime =                # Time for to move to takes (s)
Shaper = []

# Pack the parameters and initial conditions into arrays
p = [mass, cable_K, g, H, cable_C, rod_length, inertia, rod_K, rod_C, L1_begin, L2_begin, L1_end, L2_end,
     Risetime, time_begin, example_shaper.shaper]

p1 = [mass, cable_K, g, H, cable_C, rod_length, inertia, rod_K, rod_C, L1_begin, L2_begin, L1_end, L2_end,
     Risetime, time_begin, Shaper]

x0 = [X_begin, Y_begin, beta_init, x_dot_init, y_dot_init, beta_dot_init]

resp1 = odeint(eq_of_motion, x0, t, args=(p1,), atol=abserr, rtol=relerr, hmax=max_step)

resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr, hmax=max_step)


# np.savetxt("response.csv", resp1, delimiter=",")
# np.savetxt("sim_time.csv", t, delimiter=",")
# np.savetxt("response_s.csv", resp, delimiter=",")

########### PLOTTING #####################

R_amount_shaped = np.sqrt((10 - (resp[:,0] + rod_length * np.sin(resp[:,2])))**2 +
           (11 - (resp[:,1] + rod_length * np.cos(resp[:,2])))**2)

# R_amount_unshaped = np.sqrt(((resp1[:,0] + rod_length * np.sin(resp1[:,2])) - 10)**2 +
#            ((resp1[:,1] + rod_length * np.cos(resp1[:,2])) - 10.5)**2)

R_amount_unshaped = np.sqrt((10 - (resp1[:,0] + rod_length * np.sin(resp1[:,2])))**2 +
           (11 - (resp1[:,1] + rod_length * np.cos(resp1[:,2])))**2)


plt.figure(0)
# plt.subplot(211)
plt.plot(t, R_amount_shaped, label='Shaped')
plt.plot(t, R_amount_unshaped, label='Unshaped', linestyle='--')
# plt.ylim(25,0)
plt.legend()
# plt.title('R Length Change', fontsize=20)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.savefig("Shaped_vs_Unshaped.pdf")
plt.show()



# plt.figure(0)
# # plt.subplot(211)
# plt.plot(t, resp[:,1], label='Shaped')
# plt.plot(t, resp1[:,1], label='Unshaped')
# plt.ylim(25,0)
# plt.legend()
# plt.title('Y Motion')
# plt.xlabel('time (s)')
# plt.ylabel('meters')
# plt.show()

# plt.figure(1)
# plt.plot(t, resp[:,0], label='Shaped')
# plt.plot(t, resp1[:,0], label='Unshaped')
# plt.legend()
# plt.title('X Motion')
# plt.xlabel('time (s)')
# plt.ylabel('meters')
# plt.show()

# plt.figure(2)
# # plt.plot(t, np.degrees(resp[:,5]), label='Shaped')
# # plt.plot(t, np.degrees(resp1[:,5]), label='Unshaped')
# plt.plot(t, np.degrees(resp[:,2]), label='Shaped')
# plt.plot(t, np.degrees(resp1[:,2]), label='Unshaped')
# plt.legend()
# plt.ylim(-40,40)
# # plt.xlim(0,60)
# plt.title('Beta Motion')
# plt.xlabel('time (s)')
# plt.ylabel('Degrees')
# plt.savefig("Beta_Motion.pdf")
# plt.show()
