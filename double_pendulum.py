"""
Simple visualization showing how the double pendulum is sensitive
to initial conditions. 

Animation of how two double pendulums with almost similar start
position suddenly diverge.

Simulated using RK4 and sped up using numba JIT compilation.

By SLS
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, sin, cos, pi
import matplotlib.animation as animation
from time import time
from numba import jit

# Physical properties of two double pendula (P1 and P2)
m1 = 0.1                    # Mass of P1
m2 = 0.1                    # Mass of P2
l1 = 0.3                    # Length of P1
l2 = 0.3                    # Length of P2
g = 9.8                     # Gravitational acceleration
h = 0.001                   # Time step
tf = 3                      # Final time
theta1_0 = 179*pi/180       # Initial angular position of P1
theta2_0 = 180*pi/180       # Initial angular position of P2 
p1_0     = 0.0              # Initial momentum of P1
p2_0     = 0.0              # Initial momentum of P2
perturbance = 1e-4          # Disturbance to the second pendulum

# Energy
E_ref = m1*l1*g + m2*(l1+l2)*g
n = int(tf/h)           # Number of time steps

@jit(nopython=True)
def slope(y):
    """Calculating the slope at y"""
    theta1 = y[0]
    theta2 = y[1]
    p1 = y[2]
    p2 = y[3]
    A1 = (p1*p2*sin(theta1 - theta2))/(l1*l2*(m1+m2*sin(theta1 - theta2)**2))
    A2 = 1/(2*l1**2*l2**2*(m1+m2*sin(theta1-theta2)**2)**2)*(p1**2*m2*l2**2 - 2*p1*p2*m2*l1*l2*cos(theta1-theta2) + p2**2*(m1+m2)*l1**2)*sin(2*(theta1-theta2))
    theta_d1 = (p1*l2 - p2*l1*cos(theta1-theta2))/(l1**2*l2*(m1+m2*sin(theta1-theta2)**2))
    theta_d2 = (p2*(m1+m2)*l1 -p1*m2*l2*cos(theta1-theta2))/(m2*l1*l2**2*(m1+m2*sin(theta1-theta2)**2))
    p1 = -(m1+m2)*g*l1*sin(theta1) - A1 + A2
    p2 = -m2*g*l2*sin(theta2) + A1 - A2
    return np.array([theta_d1, theta_d2, p1, p2])


@jit(nopython=True)
def RK4(y, steps):
    """Performing a single step using RK4"""
    for _ in range(steps):
        y1 = h*slope(y)
        y2 = h*slope(y + 0.5*y1)
        y3 = h*slope(y + 0.5*y2)
        y4 = h*slope(y+y3)
        y += 1/6*(y1 + 2*y2 + 2*y3 + y4)
    return y

@jit(nopython=True)
def calc_energy(y):
    kinetic_energy = y[2]**2/(2*m1) + y[3]**2/(2*m2)
    potential_energy = m1*g*l1*cos(y[0]) + m2*g*(l1*cos(y[0]) + l2*cos(y[1]))
    return kinetic_energy + potential_energy

@jit(nopython=True)
def get_positions(y):
    x1 = l1*sin(y[0])
    y1 = -l1*cos(y[0])
    x2 = x1 + l2*sin(y[1])
    y2 = y1 - l2*cos(y[1])
    return [0, x1, x2], [0, y1, y2]

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', xlim=(-0.75, 0.75), ylim=(-0.75, 0.75))

line1, = ax.plot([], [], 'o-')
line2, = ax.plot([], [], 'o-')
energy_label = ax.text(0.8, 0.9, '', transform=ax.transAxes)
time_label = ax.text(0.1, 0.9, '', transform=ax.transAxes)

def init(): # Initialise animation
    line1.set_data([], [])
    line2.set_data([], [])
    energy_label.set_text('')
    time_label.set_text('')
    return line1, line2, energy_label, time_label

y1 = np.array([theta1_0, theta2_0, p1_0, p2_0])
y2 = np.array([theta1_0+perturbance, theta2_0, p1_0, p2_0])
def animate(i): # Main function to update animation
    global y1, y2
    steps = 10
    line1.set_data(get_positions(y1))
    line2.set_data(get_positions(y2))
    time_label.set_text(i)
    #energy_label.set_text("{0:.5f}".format(calc_energy(y)))
    y1 = RK4(y1, steps)
    y2 = RK4(y2, steps)
    return line1, line2, energy_label, time_label


ani = animation.FuncAnimation(fig, animate, frames=2000, interval=1, init_func=init)
Writer = animation.writers["ffmpeg"]
writer = Writer(fps=60, metadata=dict(artist="Me"), bitrate=1800)
ani.save("double_pendulum.gif", writer=writer)

#plt.title(r"Double pendulum (RK4, $\Delta t$ = {0:.1f})".format(h), fontsize=12)
plt.show()
