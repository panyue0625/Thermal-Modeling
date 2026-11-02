import sys

import numpy as np
import matplotlib.pyplot as plt
import ufl

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


# --------------------------------------
# 1. Parameters
# --------------------------------------
length = 1.0
thermal_k = 1.0
T0 = 300.0

q0 = 100.0
sigma = 0.08

#N = 40  # Grid intervals in each direction
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

# --------------------------------------
# 2. Create 2D triangular mesh
# --------------------------------------
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [
        np.array([0.0, 0.0]),
        np.array([length, length])
    ],
    [N, N],
    cell_type=mesh.CellType.triangle
)

V = fem.functionspace(domain, ("Lagrange", 1))


# --------------------------------------
# 3. Dirichlet boundary conditions
# --------------------------------------
def boundary(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], length)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], length)
    )

boundary_dofs = fem.locate_dofs_geometrical(
    V, boundary
)

bc = fem.dirichletbc(
    default_scalar_type(T0),
    boundary_dofs,
    V
)


# --------------------------------------
# 4. Define Gaussian heat source
# --------------------------------------
q = fem.Function(V)

def gaussian_heat(x):
    r2 = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
    return q0 * np.exp(-r2 / (2 * sigma**2))

q.interpolate(gaussian_heat)


# --------------------------------------
# 5. FEM weak form
# --------------------------------------
T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k = fem.Constant(
    domain, default_scalar_type(thermal_k)
)

a = k * ufl.dot(
    ufl.grad(T), ufl.grad(v)
) * ufl.dx

L_form = q * v * ufl.dx


# --------------------------------------
# 6. Solve
# --------------------------------------
problem = LinearProblem(
    a,
    L_form,
    bcs=[bc],
    petsc_options_prefix="thermal_2d_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

Th = problem.solve()


# --------------------------------------
# 7. Extract data (single MPI process)
# --------------------------------------
coords = V.tabulate_dof_coordinates()

x = coords[:, 0]
y = coords[:, 1]
temperature = Th.x.array.real

print("Number of grid intervals:", N)
print("Maximum temperature:", temperature.max())
print("Minimum temperature:", temperature.min())

with open("convergence_2d.csv", "a") as f:
    f.write(f"{N},{length/N},{temperature.max():.10f}\n")
# --------------------------------------
# 8. Plot 2D temperature map
# --------------------------------------
plt.figure(figsize=(7, 6))

contour = plt.tricontourf(
    x, y, temperature,
    levels=40,
    cmap="hot"
)

plt.colorbar(contour, label="Temperature (K)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Steady-State FEM Thermal Simulation")
plt.axis("equal")
plt.tight_layout()

# --------------------------------------
# 8. Plot only when N = 20
# --------------------------------------

if N == 20:
    plt.figure(figsize=(7, 6))

    contour = plt.tricontourf(
        x, y, temperature,
        levels=40,
        cmap="hot"
    )

    plt.colorbar(contour, label="Temperature (K)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D FEM Thermal Simulation (N=20)")
    plt.axis("equal")
    plt.tight_layout()

    plt.show()