"""
Visualizing the classical chaotic differential equations by
Edward Lorenz [1], known as the Lorenz attractor [2].

Integrated using 4th order Runge-Kutta [3]

By SLS

[1] [Lorenz (1962) Deterministic Nonperiodic Flow](https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2)
[2] [Wikipedia: Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system)
[3] [Wikipedia: Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants (from Lorenz himself)
sigma = 10
beta = 8/3
rho = 28

# Number of time steps
h = 0.01
tf = 80
N = int(tf/h)

def f(t, u):
    """Calculate slope"""
    return np.array([
        sigma*(u[1] - u[0]),
        u[0]*(rho - u[2]) - u[1],
        u[0]*u[1] - beta*u[2]
        ])

# Integrate using RK4
y = np.zeros((3, N))
y[:,0] = [1, 1, 1]
t = np.zeros(N)
for i in range(1,N):
    k1 = h * f(t[i-1], y[:, i-1])
    k2 = h * f(t[i-1] + h/2, y[:, i-1] + k1/2)
    k3 = h * f(t[i-1] + h/2, y[:, i-1] + k2/2)
    k4 = h * f(t[i-1] + h, y[:, i-1] + k3)

    y[:, i] = y[:, i-1] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

    t[i] = t[i-1] + h

# Plot static figure and rotate plot for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y[0, :], y[1, :], y[2, :], color="red", linewidth=0.2)

# Save figure
plt.savefig("lorenz.png", bbox_inches="tight")

# Static plot
plt.show()

# Animation with rotating plot for better visualization
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)