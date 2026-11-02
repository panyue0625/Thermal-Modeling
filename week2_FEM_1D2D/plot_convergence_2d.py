
import numpy as np
import matplotlib.pyplot as plt

# Read computed results
data = np.loadtxt("convergence_2d.csv", delimiter=",")

N = data[:, 0]
h = data[:, 1]
Tmax = data[:, 2]

# Print convergence table
print("N       Tmax (K)       Change (K)")
for i in range(len(N)):
    if i == 0:
        print(f"{int(N[i]):3d}     {Tmax[i]:.8f}      ---")
    else:
        change = abs(Tmax[i] - Tmax[i-1])
        print(f"{int(N[i]):3d}     {Tmax[i]:.8f}      {change:.8e}")

# Plot peak temperature against mesh size
plt.figure(figsize=(7, 5))

plt.plot(h, Tmax, "o-")

plt.xlabel("Mesh Size h")
plt.ylabel("Maximum Temperature (K)")
plt.title("2D FEM: Peak Temperature Convergence")
plt.grid(True)
plt.tight_layout()

plt.savefig("convergence_2d.png", dpi=200)
plt.show()
