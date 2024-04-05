import numpy as np
import matplotlib.pyplot as plt

# Given image gradients and intensity differences
Ix = np.array([10.75, 17.50])
Iy = np.array([20.25, 17.00])
It = np.array([11.25, 6.00])

# Calculate the difference vector
diff_vector = np.array([Ix[1] - Ix[0], Iy[1] - Iy[0]])

# Plot displacement vectors
plt.figure()
plt.quiver([0, 0], [0, 0], Ix, Iy, color=['r', 'b'], angles='xy', scale_units='xy', scale=1)
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel('Ix')
plt.ylabel('Iy')
plt.title('Optical Flow (Graphical Method)')
plt.grid()

# Draw the difference vector
plt.quiver(Ix[0], Iy[0], diff_vector[0], diff_vector[1],
           color='g', angles='xy', scale_units='xy', scale=1)

plt.show()


def lucas_flow(Ix, Iy):
    # Construct the coefficient matrix A
    A = np.vstack((Ix, Iy)).T

    # Compute the pseudo-inverse of A
    pseudo_inverse_A = np.linalg.pinv(A)

    # Compute the optical flow
    flow = np.dot(pseudo_inverse_A, np.eye(2))

    return flow

flow = lucas_flow(Ix, Iy)

plt.figure()
plt.quiver(0, 0, flow[0, 0], flow[0, 1], color='r', angles='xy', scale_units='xy', scale=1)
plt.xlabel('X Displacement')
plt.ylabel('Y Displacement')
plt.title('Optical Flow')

# Add arrow to the line plot

plt.grid()
plt.show()

