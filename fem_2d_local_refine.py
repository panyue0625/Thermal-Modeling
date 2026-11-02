import numpy as np
import matplotlib.pyplot as plt
import ufl

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

# --------------------------------------
# 1. Physical / model parameters
# --------------------------------------
length = 1.0
thermal_k = 1.0
T0 = 300.0

# Move heat source away from center
xc = 0.68
yc = 0.38

q0 = 100.0
sigma = 0.08

# Base mesh size
base_N = 20

# Local refinement radii around hotspot
refine_radii = [0.22, 0.12, 0.06]

# --------------------------------------
# 2. Build locally refined mesh
# --------------------------------------
def build_local_refined_mesh(base_N, xc, yc, refine_radii):
    # Start from a coarse uniform mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([length, length])],
        [base_N, base_N],
        cell_type=mesh.CellType.triangle
    )

    tdim = domain.topology.dim
    edge_dim = 1

    for radius in refine_radii:
        # Make sure edges and cell-edge connectivity exist
        domain.topology.create_entities(edge_dim)
        domain.topology.create_connectivity(tdim, edge_dim)

        # Get all local cells
        num_cells = (
            domain.topology.index_map(tdim).size_local
            + domain.topology.index_map(tdim).num_ghosts
        )
        cells = np.arange(num_cells, dtype=np.int32)

        # Compute cell midpoints
        mids = mesh.compute_midpoints(domain, tdim, cells)

        # Mark cells whose midpoint is near the hotspot
        dist2 = (mids[:, 0] - xc) ** 2 + (mids[:, 1] - yc) ** 2
        marked_cells = cells[dist2 < radius ** 2]

        if len(marked_cells) == 0:
            continue

        # Convert marked cells to incident edges
        marked_edges = mesh.compute_incident_entities(
            domain.topology, marked_cells, tdim, edge_dim
        )

        # Refine only those edges
        domain, _, _ = mesh.refine(domain, marked_edges)

    return domain

domain = build_local_refined_mesh(base_N, xc, yc, refine_radii)

# --------------------------------------
# 3. Function space
# --------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))

# --------------------------------------
# 4. Dirichlet boundary condition
# --------------------------------------
def boundary(x):
    return (
        np.isclose(x[0], 0.0)
        | np.isclose(x[0], length)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], length)
    )

boundary_dofs = fem.locate_dofs_geometrical(V, boundary)

bc = fem.dirichletbc(
    default_scalar_type(T0),
    boundary_dofs,
    V
)

# --------------------------------------
# 5. Gaussian heat source (off-center)
#    Use UFL directly instead of interpolation
# --------------------------------------
x = ufl.SpatialCoordinate(domain)

q_expr = q0 * ufl.exp(
    -(
        (x[0] - xc) ** 2
        + (x[1] - yc) ** 2
    ) / (2 * sigma ** 2)
)

# --------------------------------------
# 6. Weak form
# --------------------------------------
T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k = fem.Constant(domain, default_scalar_type(thermal_k))

a = k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
L_form = q_expr * v * ufl.dx

# --------------------------------------
# 7. Solve
# --------------------------------------
problem = LinearProblem(
    a,
    L_form,
    bcs=[bc],
    petsc_options_prefix="thermal_2d_local_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

Th = problem.solve()

# --------------------------------------
# 8. Extract solution for plotting
# --------------------------------------
coords = V.tabulate_dof_coordinates()
x_dof = coords[:, 0]
y_dof = coords[:, 1]
temperature = Th.x.array.real

# Mesh statistics
tdim = domain.topology.dim
num_cells_global = domain.topology.index_map(tdim).size_global
num_vertices_global = domain.topology.index_map(0).size_global

print("Off-center hotspot at:", (xc, yc))
print("Base mesh intervals:", base_N)
print("Local refinement radii:", refine_radii)
print("Number of cells:", num_cells_global)
print("Number of vertices:", num_vertices_global)
print("Maximum temperature:", temperature.max())
print("Minimum temperature:", temperature.min())

# --------------------------------------
# 9. Plot temperature field
# --------------------------------------
plt.figure(figsize=(7, 6))

contour = plt.tricontourf(
    x_dof, y_dof, temperature,
    levels=40,
    cmap="hot"
)

plt.colorbar(contour, label="Temperature (K)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D FEM with Local Mesh Refinement")
plt.axis("equal")
plt.tight_layout()

plt.savefig("fem_2d_local_refine_temperature.png", dpi=200)
plt.show()

# --------------------------------------
# 10. Plot actual FEM mesh
# --------------------------------------
from dolfinx.plot import vtk_mesh
from matplotlib.tri import Triangulation

topology, cell_types, geometry = vtk_mesh(
    domain, domain.topology.dim
)

# Extract triangular cell connectivity
triangles = topology.reshape(-1, 4)[:, 1:4]

triang = Triangulation(
    geometry[:, 0],
    geometry[:, 1],
    triangles
)

fig, ax = plt.subplots(figsize=(8, 8))

ax.triplot(
    triang,
    color="steelblue",
    linewidth=0.35
)

ax.scatter(
    xc, yc,
    color="red",
    s=65,
    label="Heat Source Center"
)

ax.set_aspect("equal")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(
    f"Locally Refined FEM Mesh ({num_cells_global} cells)"
)

ax.legend()
plt.tight_layout()

plt.savefig("local_refined_mesh.png", dpi=250)
plt.show()