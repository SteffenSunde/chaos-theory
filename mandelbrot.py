# Very slow and simple way of visualizing Mandelbrot set
# Ref. https://en.wikipedia.org/wiki/Mandelbrot_set
# By SLS

import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit

# Problem resolution
N = 1000

# Plotting limits
a, b, c, d = -2, 1, -1, 1

@jit(nopython=True)
def f(real, imag, max_iter=200, limit=1e6):
    """
    Returns true if the iteration of the quadratic map 
    $$z_{n+1} = z_n^2 + c$$ is bounded. Returns false
    otherwise.
    """
    z = complex(0,0)
    c = complex(real, imag)
    for i in range(1, max_iter):
        z = z*z + c
        if np.abs(z) > limit:
            return i
    return -1

@jit(nopython=True)
def calc_heights(a, b, c, d, N):
    """
    Returns a N by N array of values in x[a,b] and y[c,d]
    representing the number of iterations for the function
    f to diverge.
    """
    heights = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x = a + i*(b-a)/N
            y = c + j*(d-c)/N
            iterations = f(x, y)
            if iterations > 0:
                heights[j, i] = iterations
    return heights

# Calculate for each "pixel" in plot
points_x = np.linspace(a, b, N)
points_y = np.linspace(c, d, N)

start_time = time.time()
points_z = calc_heights(a, b, c, d, N)
elapsed_time = time.time() - start_time
print("Finished on resolution of {0}x{0}. Elapsed time: {1:.1f} sec.".format(N, elapsed_time))

# Plot result
fig, ax = plt.subplots()
ax.contourf(points_x, points_y, points_z)
ax.set_aspect("equal")
ax.set_xlabel("Re[c]")
ax.set_ylabel("Im[c]")
plt.savefig("example.png")
plt.show()