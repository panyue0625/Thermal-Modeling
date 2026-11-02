
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import ufl
import time

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

# =====================================
# 1. Physical parameters
# =====================================

length = 1.0
thermal_k = 1.0
T0 = 300.0

# Off-center Gaussian heat source
xc = 0.68
yc = 0.38
q0 = 100.0
sigma = 0.08

# Mesh settings
uniform_N = [20, 40, 80, 160]
base_N = 20
refine_radii = [0.22, 0.12, 0.06]


# =====================================
# 2. Create uniform mesh
# =====================================

def create_uniform(N):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [
            np.array([0.0, 0.0]),
            np.array([length, length])
        ],
        [N, N],
        cell_type=mesh.CellType.triangle
    )


# =====================================
# 3. Create locally refined mesh
# =====================================

def create_local_mesh():
    domain = create_uniform(base_N)

    tdim = domain.topology.dim
    edge_dim = 1

    for radius in refine_radii:

        domain.topology.create_entities(edge_dim)
        domain.topology.create_connectivity(
            tdim, edge_dim
        )

        num_cells = (
            domain.topology.index_map(tdim).size_local
            + domain.topology.index_map(tdim).num_ghosts
        )

        cells = np.arange(
            num_cells, dtype=np.int32
        )

        mids = mesh.compute_midpoints(
            domain, tdim, cells
        )

        dist2 = (
            (mids[:, 0] - xc)**2
            + (mids[:, 1] - yc)**2
        )

        marked_cells = cells[
            dist2 < radius**2
        ]

        if len(marked_cells) == 0:
            continue

        marked_edges = mesh.compute_incident_entities(
            domain.topology,
            marked_cells,
            tdim,
            edge_dim
        )

        domain, _, _ = mesh.refine(
            domain, marked_edges
        )

    return domain


# =====================================
# 4. FEM thermal solver
# =====================================

def solve_thermal(domain):

    start = time.perf_counter()

    V = fem.functionspace(
        domain, ("Lagrange", 1)
    )

    # Boundary conditions
    def boundary(x):
        return (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], length)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], length)
        )

    dofs = fem.locate_dofs_geometrical(
        V, boundary
    )

    bc = fem.dirichletbc(
        default_scalar_type(T0), dofs, V
    )

    # Gaussian source using UFL
    x = ufl.SpatialCoordinate(domain)

    q_expr = q0 * ufl.exp(
        -(
            (x[0] - xc)**2
            + (x[1] - yc)**2
        ) / (2 * sigma**2)
    )

    # Weak form
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    k = fem.Constant(
        domain,
        default_scalar_type(thermal_k)
    )

    a = k * ufl.dot(
        ufl.grad(T), ufl.grad(v)
    ) * ufl.dx

    L_form = q_expr * v * ufl.dx

    # Solve
    problem = LinearProblem(
        a,
        L_form,
        bcs=[bc],
        petsc_options_prefix="mesh_compare_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        }
    )

    Th = problem.solve()

    elapsed = time.perf_counter() - start

    # Results
    coords = V.tabulate_dof_coordinates()
    temperature = Th.x.array.real

    num_cells = (
        domain.topology.index_map(
            domain.topology.dim
        ).size_global
    )

    result = {
        "cells": num_cells,
        "Tmax": float(np.max(temperature)),
        "Tmin": float(np.min(temperature)),
        "time": elapsed,
        "coords": coords,
        "temperature": temperature,
        "domain": domain
    }

    return result


# =====================================
# 5. Uniform mesh experiments
# =====================================

results = []

print("\n=== UNIFORM MESH ===")

for N in uniform_N:

    domain = create_uniform(N)
    result = solve_thermal(domain)

    result["name"] = f"Uniform N={N}"
    results.append(result)

    print(
        f"N={N:3d} | "
        f"Cells={result['cells']:6d} | "
        f"Tmax={result['Tmax']:.9f} K | "
        f"Time={result['time']:.3f} s"
    )


# =====================================
# 6. Local refinement experiment
# =====================================

print("\n=== LOCAL REFINEMENT ===")

local_domain = create_local_mesh()
local_result = solve_thermal(local_domain)
local_result["name"] = "Local Refinement"

results.append(local_result)

print(
    f"Cells={local_result['cells']} | "
    f"Tmax={local_result['Tmax']:.9f} K | "
    f"Time={local_result['time']:.3f} s"
)


# =====================================
# 7. Compare against reference
# =====================================

# Finest uniform mesh is our numerical
# reference, not the exact solution.
reference = results[len(uniform_N) - 1]["Tmax"]

print("\n=== COMPARISON ===")
print(f"Reference Tmax: {reference:.9f} K")
print(
    f"{'Method':22s} {'Cells':>8s} "
    f"{'Tmax':>14s} {'Error':>14s}"
)

for r in results:

    error = abs(r["Tmax"] - reference)
    r["error"] = error

    print(
        f"{r['name']:22s} "
        f"{r['cells']:8d} "
        f"{r['Tmax']:14.9f} "
        f"{error:14.8e}"
    )


# =====================================
# 8. Plot local refined FEM mesh
# =====================================

topology, cell_types, geometry = vtk_mesh(
    local_domain, local_domain.topology.dim
)

triangles = topology.reshape(-1, 4)[:, 1:4]

triangulation = mtri.Triangulation(
    geometry[:, 0],
    geometry[:, 1],
    triangles
)

fig, ax = plt.subplots(figsize=(8, 8))

ax.triplot(
    triangulation,
    color="steelblue",
    linewidth=0.35
)

ax.scatter(
    xc, yc,
    color="red",
    s=70,
    label="Heat Source Center",
    zorder=5
)

ax.set_aspect("equal")
ax.set_xlim(0, length)
ax.set_ylim(0, length)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title(
    f"Local Refinement "
    f"({local_result['cells']} cells)"
)

ax.legend()

plt.tight_layout()
plt.savefig("compare_local_mesh.png", dpi=250)
plt.close()


# =====================================
# 9. Plot temperature field
# =====================================

coords = local_result["coords"]
temperature = local_result["temperature"]

fig, ax = plt.subplots(figsize=(7, 6))

contour = ax.tricontourf(
    coords[:, 0],
    coords[:, 1],
    temperature,
    levels=40,
    cmap="hot"
)

fig.colorbar(
    contour, ax=ax,
    label="Temperature (K)"
)

ax.scatter(
    xc, yc,
    marker="+",
    color="cyan",
    s=70
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_title("Off-Center Hotspot: Local FEM")

plt.tight_layout()
plt.savefig(
    "compare_local_temperature.png", dpi=200
)
plt.close()


# =====================================
# 10. Plot error vs number of cells
# =====================================

uniform_results = results[:len(uniform_N)]

# Reference point has zero error;
# omit it from the logarithmic plot.
plot_results = uniform_results[:-1]

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(
    [r["cells"] for r in plot_results],
    [r["error"] for r in plot_results],
    "o-",
    label="Uniform Mesh"
)

ax.scatter(
    local_result["cells"],
    local_result["error"],
    color="red",
    s=100,
    marker="*",
    label="Local Refinement",
    zorder=5
)

ax.set_xlabel("Number of Elements")
ax.set_ylabel("Peak Temperature Difference (K)")
ax.set_title("FEM Mesh Efficiency Comparison")
ax.grid(True, which="both")
ax.legend()

plt.tight_layout()
plt.savefig(
    "compare_mesh_efficiency.png", dpi=200
)
plt.close()

print("\nPlots saved successfully.")
