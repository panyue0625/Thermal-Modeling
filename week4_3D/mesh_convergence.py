
"""
Week 4 - 3D chip thermal model: mesh convergence study (FEniCSx / dolfinx >= 0.10)
 
Same model as chip3d_single_block_energy_check.py (one CORE_LEFT block, Robin backside,
adiabatic sides). The mesh is refined uniformly in x, y AND z together:
    10x10x2 -> 20x20x4 -> 40x40x8 -> 80x80x16
 
For every mesh we record
    - T_probe       : temperature at the CORE_LEFT centre on the DEVICE_FACE
                      (a fixed point -> clean convergence, unlike the node maximum)
    - T_max         : maximum nodal temperature
    - side_flux     : heat leaving the 4 adiabatic sides computed from -k grad(T).n
                      (exactly 0 in reality -> any value is pure discretisation error)
    - avg_error     : DEVICE_FACE average vs. hand formula (should stay ~1e-13 on every mesh)
    - runtime
 
Outputs:
    mesh_convergence_results.csv
    mesh_convergence_plots.png
"""
import csv
import time
 
import numpy as np
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
 
# ------------------------------------------------------------------
# 1. Parameters (SI units, identical to the energy-check script)
# ------------------------------------------------------------------
MM = 1e-3
W_PER_CM2 = 1e4
 
CHIP_LENGTH_X = 10 * MM
CHIP_LENGTH_Y = 10 * MM
CHIP_THICKNESS = 0.5 * MM
 
K_SILICON = 150.0
H_HEATSINK = 1.0e4
T_AMBIENT = 0.0
 
POWER_BLOCKS = {
    "CORE_LEFT": (1 * MM, 4 * MM, 5 * MM, 9 * MM, 150 * W_PER_CM2),
}
BACKGROUND_POWER_DENSITY = 0.0
 
# Probe point: centre of CORE_LEFT on the DEVICE_FACE
PROBE_POINT = np.array([2.5 * MM, 7.0 * MM, 0.0])
 
# Meshes (cells in x, y, z). Every x-y spacing is a divisor of 1 mm,
# so block edges always sit on mesh lines.
MESH_LIST = [(10, 10, 2), (20, 20, 4), (40, 40, 8), (80, 80, 16)]
 
# Face markers
DEVICE_FACE, BACKSIDE_FACE = 1, 2
SIDE_X_MIN, SIDE_X_MAX, SIDE_Y_MIN, SIDE_Y_MAX = 3, 4, 5, 6
SIDE_FACES = (SIDE_X_MIN, SIDE_X_MAX, SIDE_Y_MIN, SIDE_Y_MAX)
 
comm = MPI.COMM_WORLD
 
 
# ------------------------------------------------------------------
# 2. One full solve on a given mesh
# ------------------------------------------------------------------
def solve_chip(cells_x, cells_y, cells_z):
    t_start = time.perf_counter()
 
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([CHIP_LENGTH_X, CHIP_LENGTH_Y, CHIP_THICKNESS])],
        [cells_x, cells_y, cells_z],
        cell_type=mesh.CellType.tetrahedron,
    )
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
 
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
    order = np.argsort(facet_ids)
    face_tags = mesh.meshtags(domain, fdim, facet_ids[order], facet_vals[order])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=face_tags)
 
    # Power map
    x = ufl.SpatialCoordinate(domain)
    power_density = fem.Constant(domain, default_scalar_type(BACKGROUND_POWER_DENSITY))
    for (x0, x1, y0, y1, p) in POWER_BLOCKS.values():
        inside = ufl.And(ufl.And(ufl.ge(x[0], x0), ufl.le(x[0], x1)),
                         ufl.And(ufl.ge(x[1], y0), ufl.le(x[1], y1)))
        power_density = power_density + ufl.conditional(inside, p - BACKGROUND_POWER_DENSITY, 0.0)
 
    # Weak form
    V = fem.functionspace(domain, ("Lagrange", 1))
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = fem.Constant(domain, default_scalar_type(K_SILICON))
    h = fem.Constant(domain, default_scalar_type(H_HEATSINK))
    T_amb = fem.Constant(domain, default_scalar_type(T_AMBIENT))
    F = (k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
         + h * (T - T_amb) * v * ds(BACKSIDE_FACE)
         - power_density * v * ds(DEVICE_FACE))
 
    # CG + algebraic multigrid: scales to the 80x80x16 mesh (~110k unknowns),
    # where a direct LU solve gets slow and memory hungry.
    problem = LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=[],
        petsc_options_prefix=f"chip3d_conv_{cells_x}_{cells_z}_",
        petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-12},
    )
    T_solution = problem.solve()
    runtime = time.perf_counter() - t_start
 
    # Post-processing
    n = ufl.FacetNormal(domain)
 
    def integrate(expr):
        return comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)
 
    chip_area = CHIP_LENGTH_X * CHIP_LENGTH_Y
    total_power = integrate(power_density * ds(DEVICE_FACE))
    side_flux = sum(integrate(-k * ufl.dot(ufl.grad(T_solution), n) * ds(m)) for m in SIDE_FACES)
    T_device_avg = integrate(T_solution * ds(DEVICE_FACE)) / chip_area
    expected_avg = T_AMBIENT + total_power * (CHIP_THICKNESS / (K_SILICON * chip_area)
                                              + 1.0 / (H_HEATSINK * chip_area))
    avg_error = abs(T_device_avg - expected_avg) / (expected_avg - T_AMBIENT)
    T_max = comm.allreduce(T_solution.x.array.max(), op=MPI.MAX)
 
    # Temperature at the probe point
    tree = geometry.bb_tree(domain, tdim)
    candidates = geometry.compute_collisions_points(tree, PROBE_POINT.reshape(1, 3))
    cell = geometry.compute_colliding_cells(domain, candidates, PROBE_POINT.reshape(1, 3)).links(0)[0]
    T_probe = T_solution.eval(PROBE_POINT.reshape(1, 3), np.array([cell], dtype=np.int32))[0]
 
    return {
        "mesh": f"{cells_x}x{cells_y}x{cells_z}",
        "h_xy_mm": CHIP_LENGTH_X / cells_x / MM,
        "cells": domain.topology.index_map(tdim).size_global,
        "unknowns": V.dofmap.index_map.size_global,
        "runtime_s": runtime,
        "T_probe": T_probe - T_AMBIENT,
        "T_max": T_max - T_AMBIENT,
        "side_flux_W": side_flux,
        "side_flux_pct": side_flux / total_power * 100,
        "avg_error": avg_error,
    }
 
 
# ------------------------------------------------------------------
# 3. Run all meshes
# ------------------------------------------------------------------
results = []
for cells in MESH_LIST:
    if comm.rank == 0:
        print(f"solving {cells} ...", flush=True)
    results.append(solve_chip(*cells))
 
# Observed convergence order from three consecutive meshes (refinement ratio 2):
#   p = log2( |f(h) - f(h/2)| / |f(h/2) - f(h/4)| )
def observed_order(values):
    orders = [np.nan, np.nan]
    for i in range(2, len(values)):
        d1, d2 = abs(values[i - 2] - values[i - 1]), abs(values[i - 1] - values[i])
        orders.append(np.log2(d1 / d2) if d2 > 0 else np.nan)
    return orders
 
order_probe = observed_order([r["T_probe"] for r in results])
order_side = [np.nan] + [np.log2(abs(results[i - 1]["side_flux_W"]) / abs(results[i]["side_flux_W"]))
                         for i in range(1, len(results))]
 
if comm.rank == 0:
    print()
    header = (f"{'mesh':>10s} {'h_xy[mm]':>8s} {'cells':>9s} {'unknowns':>9s} {'time[s]':>8s} "
              f"{'T_probe':>9s} {'order':>6s} {'T_max':>9s} {'side %':>8s} {'order':>6s} {'avg err':>9s}")
    print(header)
    print("-" * len(header))
    for r, p1, p2 in zip(results, order_probe, order_side):
        print(f"{r['mesh']:>10s} {r['h_xy_mm']:8.3f} {r['cells']:9d} {r['unknowns']:9d} {r['runtime_s']:8.2f} "
              f"{r['T_probe']:9.4f} {p1:6.2f} {r['T_max']:9.4f} {r['side_flux_pct']:8.3f} {p2:6.2f} "
              f"{r['avg_error']:9.1e}")
 
    with open("mesh_convergence_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
 
    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    h_vals = np.array([r["h_xy_mm"] for r in results])
    T_probe_vals = np.array([r["T_probe"] for r in results])
    side_vals = np.abs([r["side_flux_pct"] for r in results])
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
 
    axes[0].plot(h_vals, T_probe_vals, "o-", label="T at CORE_LEFT centre")
    axes[0].plot(h_vals, [r["T_max"] for r in results], "s--", label="max nodal T")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("x-y cell size h (mm)   [finer →]")
    axes[0].set_ylabel("T - T_ambient (°C)")
    axes[0].set_title("Peak temperature vs. mesh")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
 
    # Successive differences |T(h) - T(h/2)| -> slope = convergence order
    diffs = np.abs(np.diff(T_probe_vals))
    axes[1].loglog(h_vals[1:], diffs, "o-", label="|T_probe(h) - T_probe(2h)|")
    axes[1].loglog(h_vals[1:], diffs[0] * (h_vals[1:] / h_vals[1]) ** 2, "k--", label="slope 2 (reference)")
    axes[1].loglog(h_vals[1:], diffs[0] * (h_vals[1:] / h_vals[1]) ** 1, "k:", label="slope 1 (reference)")
    axes[1].set_xlabel("h (mm)")
    axes[1].set_ylabel("change in T_probe (°C)")
    axes[1].set_title("Convergence of the probe temperature")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
 
    axes[2].loglog(h_vals, side_vals, "o-", label="|spurious side flux|")
    axes[2].loglog(h_vals, side_vals[0] * (h_vals / h_vals[0]) ** 1, "k:", label="slope 1 (reference)")
    axes[2].set_xlabel("h (mm)")
    axes[2].set_ylabel("% of total power")
    axes[2].set_title("Gradient-based side flux (true value = 0)")
    axes[2].legend()
    axes[2].grid(alpha=0.3, which="both")
 
    fig.tight_layout()
    fig.savefig("mesh_convergence_plots.png", dpi=200)
    plt.show()