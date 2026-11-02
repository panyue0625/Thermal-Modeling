
import numpy as np
import matplotlib.pyplot as plt
import ufl

from mpi4py import MPI

from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

# -----------------------------------
# 1. Physical parameters
# -----------------------------------
length = 1.0
thermal_k = 1.0
heat_q = 10.0
T0 = 300.0

# -----------------------------------
# 2. Create 1D mesh
# -----------------------------------
N = 20

domain = mesh.create_interval(
    MPI.COMM_WORLD,
    N,
    [0.0, length]
)

# Linear Lagrange finite elements
V = fem.functionspace(domain, ("Lagrange", 1))

# -----------------------------------
# 3. Boundary conditions
# -----------------------------------
def boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(
        x[0], length
    )

boundary_dofs = fem.locate_dofs_geometrical(
    V, boundary
)

bc = fem.dirichletbc(
    default_scalar_type(T0),
    boundary_dofs,
    V
)

# -----------------------------------
# 4. Trial and test functions
# -----------------------------------
T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k = fem.Constant(
    domain, default_scalar_type(thermal_k)
)
q = fem.Constant(
    domain, default_scalar_type(heat_q)
)

# -----------------------------------
# 5. Weak form
# -----------------------------------
a = k * ufl.dot(
    ufl.grad(T), ufl.grad(v)
) * ufl.dx

L_form = q * v * ufl.dx

# -----------------------------------
# 6. Solve
# -----------------------------------
problem = LinearProblem(
    a,
    L_form,
    bcs=[bc],
    petsc_options_prefix="thermal_1d_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

Th = problem.solve()

# -----------------------------------
# 7. Compare with analytical solution
# -----------------------------------
coords = V.tabulate_dof_coordinates()[:, 0]
values = Th.x.array.real

order = np.argsort(coords)

x = coords[order]
T_fem = values[order]

T_exact = T0 + heat_q / (2 * thermal_k) * x * (
    length - x
)

error = np.max(np.abs(T_fem - T_exact))

print("Maximum nodal error:", error)
print("Maximum FEM temperature:", T_fem.max())

# -----------------------------------
# 8. Plot
# -----------------------------------
plt.plot(x, T_fem, "o-", label="FEM")
plt.plot(x, T_exact, "-", label="Analytical")

plt.xlabel("Position x")
plt.ylabel("Temperature (K)")
plt.title("1D Steady-State Heat Equation - FEM")
plt.legend()
plt.grid()

plt.savefig("fem_1d.png", dpi=200)
plt.show()
