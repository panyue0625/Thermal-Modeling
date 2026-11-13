"""
Week 4 - 3D steady-state chip thermal model: single power block + energy-balance check (FEniCSx / dolfinx >= 0.10)
 
Model:
    - Domain: silicon die  CHIP_LENGTH_X x CHIP_LENGTH_Y x CHIP_THICKNESS
    - DEVICE_FACE   (z = 0): transistor layer, power injected as a surface heat flux Pd(x, y)
    - BACKSIDE_FACE (z = d): heat sink, Robin  -k dT/dn = h (T - T_ambient)
    - SIDE_FACES           : adiabatic (nothing to add to the weak form)
    - No volumetric heat source inside the silicon (q = 0)
 
This script uses ONE power block (CORE_LEFT) on a uniform mesh.
 
Checks printed at the end:
    1. Energy balance per face: heat in = heat out
    2. Average DEVICE_FACE temperature rise vs. the 1D thermal-resistance formula
           dT_avg = P * (CHIP_THICKNESS / (k A) + 1 / (h A))
       (this is exact for the face average, not just an estimate)
    3. Average BACKSIDE_FACE temperature rise vs.  P / (h A)
 
All quantities are in SI units (m, W, K). Plots use mm and degC for readability.
Outputs: single_block_face_temperatures.png, single_block_vertical_profile.png
"""
import time
 
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
 
# ------------------------------------------------------------------
# 1. Parameters (SI units)
# ------------------------------------------------------------------
MM = 1e-3                      # 1 mm in metres
W_PER_CM2 = 1e4                # 1 W/cm^2 in W/m^2
 
CHIP_LENGTH_X = 10 * MM
CHIP_LENGTH_Y = 10 * MM
CHIP_THICKNESS = 0.5 * MM
 
K_SILICON = 150.0              # W/(m K)
H_HEATSINK = 1.0e4             # W/(m^2 K), effective heat-sink coefficient on the backside
T_AMBIENT = 0.0                # degC -> the solution is directly the temperature rise
 
# Power blocks on the DEVICE_FACE: name -> (x_min, x_max, y_min, y_max, power density)
# Edges are whole millimetres so they sit exactly on mesh lines (0.25 mm spacing below).
POWER_BLOCKS = {
    "CORE_LEFT": (1 * MM, 4 * MM, 5 * MM, 9 * MM, 150 * W_PER_CM2),
}
BACKGROUND_POWER_DENSITY = 0.0  # W/m^2 outside all blocks (step 1: none)
 
# Mesh: number of cells in x, y, z (x, y spacing = 0.25 mm, z spacing = 0.0625 mm)
CELLS_X, CELLS_Y, CELLS_Z = 40, 40, 8
 
# ------------------------------------------------------------------
# 2. Mesh and face tags
# ------------------------------------------------------------------
t_start = time.perf_counter()
 
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([CHIP_LENGTH_X, CHIP_LENGTH_Y, CHIP_THICKNESS])],
    [CELLS_X, CELLS_Y, CELLS_Z],
    cell_type=mesh.CellType.tetrahedron,
)
tdim = domain.topology.dim          # 3
fdim = tdim - 1                     # 2 (faces)
domain.topology.create_connectivity(fdim, tdim)
 
DEVICE_FACE, BACKSIDE_FACE = 1, 2
SIDE_X_MIN, SIDE_X_MAX, SIDE_Y_MIN, SIDE_Y_MAX = 3, 4, 5, 6
FACE_NAMES = {
    DEVICE_FACE: "DEVICE_FACE   (z=0)",
    BACKSIDE_FACE: "BACKSIDE_FACE (z=d)",
    SIDE_X_MIN: "SIDE x=0",
    SIDE_X_MAX: "SIDE x=Lx",
    SIDE_Y_MIN: "SIDE y=0",
    SIDE_Y_MAX: "SIDE y=Ly",
}
face_locators = [
    (DEVICE_FACE,   lambda x: np.isclose(x[2], 0.0)),
    (BACKSIDE_FACE, lambda x: np.isclose(x[2], CHIP_THICKNESS)),
    (SIDE_X_MIN,    lambda x: np.isclose(x[0], 0.0)),
    (SIDE_X_MAX,    lambda x: np.isclose(x[0], CHIP_LENGTH_X)),
    (SIDE_Y_MIN,    lambda x: np.isclose(x[1], 0.0)),
    (SIDE_Y_MAX,    lambda x: np.isclose(x[1], CHIP_LENGTH_Y)),
]
facet_ids, facet_vals = [], []
for marker, locator in face_locators:
    facets = mesh.locate_entities_boundary(domain, fdim, locator)
    facet_ids.append(facets)
    facet_vals.append(np.full_like(facets, marker))
facet_ids = np.hstack(facet_ids).astype(np.int32)
facet_vals = np.hstack(facet_vals).astype(np.int32)
# Edge facets belong to no face (a face is never on two planes), so ids are unique.
order = np.argsort(facet_ids)
face_tags = mesh.meshtags(domain, fdim, facet_ids[order], facet_vals[order])
 
ds = ufl.Measure("ds", domain=domain, subdomain_data=face_tags)
 
# ------------------------------------------------------------------
# 3. Power map Pd(x, y) on the DEVICE_FACE, built with ufl.conditional
# ------------------------------------------------------------------
x = ufl.SpatialCoordinate(domain)
 
 
def block(x_min, x_max, y_min, y_max, density):
    """density inside the rectangle, 0 outside (evaluated at each quadrature point)."""
    inside = ufl.And(
        ufl.And(ufl.ge(x[0], x_min), ufl.le(x[0], x_max)),
        ufl.And(ufl.ge(x[1], y_min), ufl.le(x[1], y_max)),
    )
    return ufl.conditional(inside, density - BACKGROUND_POWER_DENSITY, 0.0)
 
 
power_density = fem.Constant(domain, default_scalar_type(BACKGROUND_POWER_DENSITY))
for name, (x0, x1, y0, y1, p) in POWER_BLOCKS.items():
    power_density = power_density + block(x0, x1, y0, y1, p)
 
# ------------------------------------------------------------------
# 4. Weak form and solve
#    find T:  ∫ k ∇T·∇v dx  +  ∫_backside h (T - T_amb) v ds  =  ∫_device Pd v ds
# ------------------------------------------------------------------
V = fem.functionspace(domain, ("Lagrange", 1))
T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
 
k = fem.Constant(domain, default_scalar_type(K_SILICON))
h = fem.Constant(domain, default_scalar_type(H_HEATSINK))
T_amb = fem.Constant(domain, default_scalar_type(T_AMBIENT))
 
F = (
    k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
    + h * (T - T_amb) * v * ds(BACKSIDE_FACE)
    - power_density * v * ds(DEVICE_FACE)
)
# SIDE_FACES: adiabatic -> no term
 
problem = LinearProblem(
    ufl.lhs(F), ufl.rhs(F), bcs=[],
    petsc_options_prefix="chip3d_single_block_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
T_solution = problem.solve()
t_solve = time.perf_counter() - t_start
 
# ------------------------------------------------------------------
# 5. Post-processing
# ------------------------------------------------------------------
comm = domain.comm
n = ufl.FacetNormal(domain)
 
 
def integrate(expr):
    return comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)
 
 
chip_area = CHIP_LENGTH_X * CHIP_LENGTH_Y
total_power = integrate(power_density * ds(DEVICE_FACE))          # W
T_max = comm.allreduce(T_solution.x.array.max(), op=MPI.MAX)
T_device_avg = integrate(T_solution * ds(DEVICE_FACE)) / chip_area
T_backside_avg = integrate(T_solution * ds(BACKSIDE_FACE)) / chip_area
 
# Heat leaving through each face:  ∫ -k ∇T·n ds   (n points outward; negative = heat entering)
heat_out = {
    marker: integrate(-k * ufl.dot(ufl.grad(T_solution), n) * ds(marker))
    for marker in FACE_NAMES
}
# On the backside the Robin law gives the same quantity more accurately (no gradient needed)
heat_out_backside_robin = integrate(h * (T_solution - T_amb) * ds(BACKSIDE_FACE))
 
# Hand calculation (1D thermal resistance in series)
R_conduction = CHIP_THICKNESS / (K_SILICON * chip_area)   # K/W
R_heatsink = 1.0 / (H_HEATSINK * chip_area)               # K/W
expected_device_avg = T_AMBIENT + total_power * (R_conduction + R_heatsink)
expected_backside_avg = T_AMBIENT + total_power * R_heatsink
 
if comm.rank == 0:
    num_cells = domain.topology.index_map(tdim).size_global
    num_dofs = V.dofmap.index_map.size_global
    print("=" * 64)
    print(f"Mesh: {CELLS_X} x {CELLS_Y} x {CELLS_Z}  ->  {num_cells} tetrahedra, {num_dofs} unknowns")
    print(f"Mesh + assemble + solve time: {t_solve:.2f} s")
    print(f"Total power injected: {total_power:.3f} W")
    print(f"Peak temperature rise: {T_max - T_AMBIENT:.3f} K")
    print()
    print("Heat leaving each face (W)   [negative = entering]")
    for marker, name in FACE_NAMES.items():
        print(f"  {name:22s} {heat_out[marker]:10.4f}   ({heat_out[marker] / total_power:7.2%} of P)")
    print(f"  backside via Robin h(T-Ta) {heat_out_backside_robin:10.4f}")
    balance_error = abs(heat_out_backside_robin - total_power) / total_power
    print(f"Energy balance error (Robin out vs. P in): {balance_error:.2e}")
    print()
    print("Average temperature rise:              FEM      hand calc   rel. error")
    print(f"  DEVICE_FACE   P(d/kA + 1/hA)    {T_device_avg - T_AMBIENT:9.4f}  {expected_device_avg - T_AMBIENT:9.4f}"
          f"   {abs(T_device_avg - expected_device_avg) / (expected_device_avg - T_AMBIENT):.2e}")
    print(f"  BACKSIDE_FACE P/(hA)            {T_backside_avg - T_AMBIENT:9.4f}  {expected_backside_avg - T_AMBIENT:9.4f}"
          f"   {abs(T_backside_avg - expected_backside_avg) / (expected_backside_avg - T_AMBIENT):.2e}")
    print("=" * 64)
 
# ------------------------------------------------------------------
# 6. Plots (serial run assumed)
# ------------------------------------------------------------------
coords = V.tabulate_dof_coordinates()
temps = T_solution.x.array.real
 
# 6a. Temperature on the DEVICE_FACE and BACKSIDE_FACE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, z_value, title in [(axes[0], 0.0, "DEVICE_FACE (z = 0)"),
                           (axes[1], CHIP_THICKNESS, "BACKSIDE_FACE (z = d)")]:
    on_face = np.isclose(coords[:, 2], z_value)
    cf = ax.tricontourf(coords[on_face, 0] / MM, coords[on_face, 1] / MM, temps[on_face],
                        levels=40, cmap="hot")
    fig.colorbar(cf, ax=ax, label="T - T_ambient (°C)")
    for name, (x0, x1, y0, y1, _) in POWER_BLOCKS.items():
        ax.add_patch(plt.Rectangle((x0 / MM, y0 / MM), (x1 - x0) / MM, (y1 - y0) / MM,
                                   fill=False, ec="cyan", lw=1.2, ls="--"))
        ax.text((x0 + x1) / 2 / MM, (y0 + y1) / 2 / MM, name, color="cyan",
                ha="center", va="center", fontsize=8)
    ax.set_title(f"{title}   max = {temps[on_face].max():.2f} °C")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("single_block_face_temperatures.png", dpi=200)
 
# 6b. Temperature along z through the centre of the first block
x0, x1, y0, y1, _ = next(iter(POWER_BLOCKS.values()))
zs = np.linspace(0.0, CHIP_THICKNESS, 101)
points = np.zeros((zs.size, 3))
points[:, 0], points[:, 1], points[:, 2] = (x0 + x1) / 2, (y0 + y1) / 2, zs
tree = geometry.bb_tree(domain, tdim)
candidates = geometry.compute_collisions_points(tree, points)
colliding = geometry.compute_colliding_cells(domain, candidates, points)
keep, cells = [], []
for i in range(points.shape[0]):
    if len(colliding.links(i)) > 0:
        keep.append(i)
        cells.append(colliding.links(i)[0])
keep = np.array(keep, dtype=np.int32)
profile = T_solution.eval(points[keep], np.array(cells, dtype=np.int32)).flatten()
 
plt.figure(figsize=(7, 5))
plt.plot(zs[keep] / MM, profile, "o-", ms=3)
plt.xlabel("z (mm)   [0 = DEVICE_FACE, 0.5 = BACKSIDE_FACE]")
plt.ylabel("T - T_ambient (°C)")
plt.title("Temperature through the thickness, under the block centre")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("single_block_vertical_profile.png", dpi=200)
plt.show()