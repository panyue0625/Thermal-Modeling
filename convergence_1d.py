
import numpy as np
import matplotlib.pyplot as plt
import ufl

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem


# 1. Physical parameters
length = 1.0
thermal_k = 1.0
heat_q = 10.0
T0 = 300.0


def solve_fem(N):
    # 2. Create mesh
    domain = mesh.create_interval(
        MPI.COMM_WORLD, N, [0.0, length]
    )

    V = fem.functionspace(domain, ("Lagrange", 1))

    # 3. Boundary conditions
    def boundary(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], length)
        )

    dofs = fem.locate_dofs_geometrical(V, boundary)

    bc = fem.dirichletbc(
        default_scalar_type(T0), dofs, V
    )

    # 4. Weak form
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = fem.Constant(
        domain, default_scalar_type(thermal_k)
    )
    q = fem.Constant(
        domain, default_scalar_type(heat_q)
    )

    a = k * ufl.dot(
        ufl.grad(T), ufl.grad(v)
    ) * ufl.dx

    L_form = q * v * ufl.dx

    # 5. Solve FEM
    problem = LinearProblem(
        a, L_form,
        bcs=[bc],
        petsc_options_prefix="thermal_conv_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        }
    )

    Th = problem.solve()

    # 6. Calculate L2 error
    # Use quadratic elements for error evaluation
    W = fem.functionspace(domain, ("Lagrange", 2))

    # Represent FEM solution in higher-order space
    Th_high = fem.Function(W)
    Th_high.interpolate(Th)

    # Exact quadratic temperature solution
    T_exact = fem.Function(W)
    T_exact.interpolate(
        lambda x: T0 + heat_q / (2 * thermal_k)
        * x[0] * (length - x[0])
    )

    # Form error function before integrating
    error_func = fem.Function(W)
    error_func.x.array[:] = (
        T_exact.x.array - Th_high.x.array
    )

    error_form = fem.form(
        ufl.inner(error_func, error_func) * ufl.dx
    )

    error_local = fem.assemble_scalar(error_form)
    error_global = domain.comm.allreduce(
        error_local, op=MPI.SUM
    )

    L2_error = np.sqrt(error_global)

    return length / N, L2_error


# 7. Mesh convergence study
N_values = [10, 20, 40, 80, 160]

h_values = []
errors = []

for N in N_values:
    h, error = solve_fem(N)

    h_values.append(h)
    errors.append(error)

    print(
        f"N={N:3d} | h={h:.5f} | "
        f"L2 error={error:.8e}"
    )


# 8. Compute convergence order
print("\nObserved convergence orders:")

for i in range(1, len(N_values)):
    p = np.log(errors[i-1] / errors[i]) / np.log(
        h_values[i-1] / h_values[i]
    )

    print(
        f"{N_values[i-1]} -> {N_values[i]}: "
        f"p = {p:.4f}"
    )


# 9. Plot error versus mesh size
plt.figure()

plt.loglog(
    h_values, errors, "o-",
    label="FEM L2 Error"
)

# Reference slope h^2
reference = errors[0] * (
    np.array(h_values) / h_values[0]
) ** 2

plt.loglog(
    h_values, reference, "--",
    label="Reference O(h^2)"
)

plt.xlabel("Mesh Size h")
plt.ylabel("L2 Error")
plt.title("1D FEM Mesh Convergence")
plt.grid(True, which="both")
plt.legend()

plt.savefig("convergence_1d.png", dpi=200)
plt.show()
