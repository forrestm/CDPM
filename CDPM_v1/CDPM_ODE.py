'''
This file takes the EOMs from CDPM_EOM.py and solves them, then plots
the response. You must first run CDPM_EOM then copy and paste the equations in
in here.

March 17:
I have updated the EOM with the correct ones although I have to have the beta
k and c as negative to make beta rotate the correct way.

March 27:
All is good and well these are the final equations, I corrected the negative
sign issue.
'''

from scipy.integrate import odeint
import numpy as np
from numpy import sin, cos, sqrt
import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt


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
        sysODE : A list representing the system of equations of motion
                 as 1st order ODEs
    """
    x, y, b, x_dot, y_dot, beta_dot = w
    m, k, g, H, c, D, t, Izz, k_beta, c_beta, L_1_init, L_2_init = p

    # Create sysODE = (x', theta', x_dot', theta_dot'):
    sysODE = [x_dot,
              y_dot,
              beta_dot,
              (D*beta_dot**2*m*sin(b)/2 - D*m*(-D*g*m*sin(b)/2 - D*(D*beta_dot**2*m*sin(b)/2 - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*cos(b)/2 - D*(-D*beta_dot**2*m*cos(b)/2 - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*sin(b)/2 - 1.0*b*k_beta - beta_dot*c_beta - m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*(-beta_dot*x_dot*sin(b) + beta_dot*y_dot*cos(b))/2)/2 + m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*beta_dot*(-x_dot*sin(b) + y_dot*cos(b))/2)/2)*cos(b)/(2*(-D**2*m*sin(b)**2/4 - D**2*m*cos(b)**2/4 + D**2*m/4 + Izz)) - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))/m
              (-D*beta_dot**2*m*cos(b)/2 - D*m*(-D*g*m*sin(b)/2 - D*(D*beta_dot**2*m*sin(b)/2 - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*cos(b)/2 - D*(-D*beta_dot**2*m*cos(b)/2 - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*sin(b)/2 - 1.0*b*k_beta - beta_dot*c_beta - m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*(-beta_dot*x_dot*sin(b) + beta_dot*y_dot*cos(b))/2)/2 + m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*beta_dot*(-x_dot*sin(b) + y_dot*cos(b))/2)/2)*sin(b)/(2*(-D**2*m*sin(b)**2/4 - D**2*m*cos(b)**2/4 + D**2*m/4 + Izz)) - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))/m
              (-D*g*m*sin(b)/2 - D*(D*beta_dot**2*m*sin(b)/2 - c*x*(x_dot + y_dot)/sqrt(x**2 + y**2) + c*(H - x)*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - 1.0*k*x*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*(-H + x)*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*cos(b)/2 - D*(-D*beta_dot**2*m*cos(b)/2 - c*y*(-x_dot + y_dot)/sqrt(y**2 + (H - x)**2) - c*y*(x_dot + y_dot)/sqrt(x**2 + y**2) + g*m - 1.0*k*y*(-L_1_init + sqrt(x**2 + y**2))/sqrt(x**2 + y**2) - 1.0*k*y*(-L_2_init + sqrt(y**2 + (H - x)**2))/sqrt(y**2 + (H - x)**2))*sin(b)/2 - 1.0*b*k_beta - beta_dot*c_beta - m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*(-beta_dot*x_dot*sin(b) + beta_dot*y_dot*cos(b))/2)/2 + m*(-D*beta_dot*x_dot*sin(b)/2 + D*beta_dot*y_dot*cos(b)/2 + D*beta_dot*(-x_dot*sin(b) + y_dot*cos(b))/2)/2)/(-D**2*m*sin(b)**2/4 - D**2*m*cos(b)**2/4 + D**2*m/4 + Izz)
              ]
    return sysODE


# Set up simulation parameters
m = 1.0
k = 10.0
g = 9.81
H = 20.0
c = 1.0
D = 3.0
inertia = D**2 * (1.0/3.0) * m
Izz = inertia
k_beta = 1.0
c_beta = 0.01
L_1_init = 13.44846387
L_2_init = 13.44846387


# ODE solver parameters
abserr = 1.0e-9
relerr = 1.0e-9
max_step = 0.01
stoptime = 50.0
numpoints = 1000

# Create the time samples for the output of the ODE solver
t = np.linspace(0.0, stoptime, numpoints)

# Initial conditions
x_init = 13                        # initial position
x_dot_init = 0.0                    # initial velocity
y_init = 12.0                  # initial angle
y_dot_init = 0.0                # initial angular velocity
beta_init = 0.0                        # initial position
beta_dot_init = 0.0

# wf = np.np.sqrt(k / m1)                # forcing function frequency

# Pack the parameters and initial conditions into arrays
p = [m, k, g, H, c, D, t, Izz, k_beta, c_beta, L_1_init, L_2_init]
x0 = [x_init, y_init, beta_init, x_dot_init, y_dot_init, beta_dot_init]
resp = odeint(eq_of_motion, x0, t, args=(p,), atol=abserr, rtol=relerr,
              hmax=max_step)
# np.savetxt("response.csv", resp, delimiter=",")
# np.savetxt("sim_time.csv", t, delimiter=",")

########### PLOTTING #####################

plt.figure(0)
# plt.subplot(211)
plt.plot(t, resp[:,1], label='Unshaped')
plt.ylim(15,0)
plt.legend()
plt.title('Y Motion')
plt.xlabel('time (s)')
plt.ylabel('meters')
plt.show()

plt.figure(1)
plt.plot(t, resp[:,0], label='Unshaped')
plt.legend()
plt.ylim(20,0)
plt.title('X Motion')
plt.xlabel('time (s)')
plt.ylabel('meters')
plt.show()

plt.figure(2)
# This negative sign is to make the angle move in the correct direction
plt.plot(t, np.degrees(resp[:,2]), label='Unshaped')
plt.legend()
# plt.ylim(-15,15)
plt.title('Beta Motion')
plt.xlabel('time (s)')
plt.ylabel('Degrees')
# plt.savefig("Beta_Motion.pdf")
plt.show()

plt.figure(3)
# plt.subplot(212)
plt.plot(resp[:,0], resp[:,1])
plt.plot(resp[:,0] + 0.5/2 * np.sin(resp[:,2]) , resp[:,1] + 0.5/2 *
                                    np.cos(resp[:,2]), label='Unshaped')
plt.plot(resp[:,0] + 1/2 * np.sin(resp[:,2]) , resp[:,1] + 1/2 *
                                  np.cos(resp[:,2]), label='Unshaped')
plt.legend()
plt.title('Front View Motion')
# plt.ylim(25,0)
plt.ylim(25,0)
# plt.xlim(0,20)
plt.xlim(0,20)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
