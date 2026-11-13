"""
Week 4 - 3D chip thermal model: full power map, FEM vs. analytical (cosine-series) solution
(FEniCSx / dolfinx >= 0.10)
 
Model (Sapatnekar EE8301 slide 29):
    - Silicon die  CHIP_LENGTH_X x CHIP_LENGTH_Y x CHIP_THICKNESS
    - DEVICE_FACE   (z = 0): power density Pd(x, y) injected as a surface heat flux
    - BACKSIDE_FACE (z = d): heat sink, Robin  -k dT/dn = h (T - T_ambient)
    - SIDE_FACES           : adiabatic
 
Power map: CORE_LEFT, CORE_RIGHT, CACHE + BACKGROUND (see POWER_BLOCKS).
 
Analytical solution (separation of variables; adiabatic sides -> cosine modes):
    T(x, y, z) - T_amb = sum_m sum_n  P_mn * G_mn(z) * cos(m pi x / a) * cos(n pi y / b)
 
    P_mn  : cosine coefficients of the power map (closed form for rectangles)
    lam   = sqrt((m pi / a)^2 + (n pi / b)^2)
    G_mn(0) = (k lam + h tanh(lam d)) / (k lam (k lam tanh(lam d) + h))       DEVICE_FACE
    G_mn(d) = 1 / (k lam sinh(lam d) + h cosh(lam d))                        BACKSIDE_FACE
    m = n = 0:  G(0) = d/k + 1/h ,  G(d) = 1/h                               (1D thermal resistance)
 
What the script does:
    1. Analytical series on the DEVICE_FACE, with a truncation check (how many terms are enough)
    2. FEM on three meshes -> true error against the analytical solution (not just mesh-to-mesh)
    3. Plots for the 40x40x8 mesh: power map / FEM / analytical / FEM - analytical
    4. Heat balance and hot-spot locations
 
Outputs:
    power_map_fem_vs_analytical.png
    power_map_error_vs_mesh.png
    power_map_error_vs_mesh.csv
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
# 1. Parameters (SI units)
# ------------------------------------------------------------------
MM = 1e-3
W_PER_CM2 = 1e4
 
CHIP_LENGTH_X = 10 * MM
CHIP_LENGTH_Y = 10 * MM
CHIP_THICKNESS = 0.5 * MM
 
K_SILICON = 150.0              # W/(m K)
H_HEATSINK = 1.0e4             # W/(m^2 K)
T_AMBIENT = 0.0                # degC
 
# name -> (x_min, x_max, y_min, y_max, power density). Blocks do not overlap.
POWER_BLOCKS = {
    "CORE_LEFT":  (1 * MM, 4 * MM, 5 * MM, 9 * MM, 150 * W_PER_CM2),
    "CORE_RIGHT": (6 * MM, 9 * MM, 5 * MM, 9 * MM, 150 * W_PER_CM2),
    "CACHE":      (1 * MM, 9 * MM, 1 * MM, 4 * MM,  30 * W_PER_CM2),
}
BACKGROUND_POWER_DENSITY = 5 * W_PER_CM2     # everywhere outside the blocks
 
PROBE_POINT = np.array([2.5 * MM, 7.0 * MM, 0.0])     # CORE_LEFT centre on the DEVICE_FACE
 
MESH_LIST = [(20, 20, 4), (40, 40, 8), (80, 80, 16)]
PLOT_MESH = (40, 40, 8)
 
SERIES_TERMS = 400                       # m, n = 0 .. SERIES_TERMS-1 for the reference solution
SERIES_TERMS_CHECK = [50, 100, 200, 400, 800]
 
DEVICE_FACE, BACKSIDE_FACE = 1, 2
SIDE_X_MIN, SIDE_X_MAX, SIDE_Y_MIN, SIDE_Y_MAX = 3, 4, 5, 6
 
comm = MPI.COMM_WORLD
a, b, d = CHIP_LENGTH_X, CHIP_LENGTH_Y, CHIP_THICKNESS
 
 
# ------------------------------------------------------------------
# 2. Analytical (cosine-series) solution
# ------------------------------------------------------------------
def cosine_integral(m, lo, hi, length):
    """∫_lo^hi cos(m pi s / length) ds  (vectorised over m)."""
    m = np.asarray(m, dtype=float)
    out = np.empty_like(m)
    zero = m == 0
    out[zero] = hi - lo
    mm = m[~zero]
    out[~zero] = length / (mm * np.pi) * (np.sin(mm * np.pi * hi / length) - np.sin(mm * np.pi * lo / length))
    return out
 
 
def power_coefficients(n_terms):
    """P_mn such that Pd(x, y) = sum P_mn cos(m pi x/a) cos(n pi y/b)."""
    m = np.arange(n_terms)
    P = np.zeros((n_terms, n_terms))
    P[0, 0] = BACKGROUND_POWER_DENSITY * a * b
    for (x0, x1, y0, y1, p) in POWER_BLOCKS.values():
        P += (p - BACKGROUND_POWER_DENSITY) * np.outer(cosine_integral(m, x0, x1, a),
                                                      cosine_integral(m, y0, y1, b))
    eps = np.where(m == 0, 1.0, 2.0)
    return P * np.outer(eps, eps) / (a * b)
 
 
def modal_response(n_terms, face):
    """G_mn: temperature rise per unit power density of mode (m, n), on the given face."""
    m = np.arange(n_terms)
    lam = np.sqrt(np.add.outer((m * np.pi / a) ** 2, (m * np.pi / b) ** 2))
    G = np.empty_like(lam)
    k, h = K_SILICON, H_HEATSINK
    pos = lam > 0
    L = lam[pos]
    if face == "device":
        t = np.tanh(L * d)
        G[pos] = (k * L + h * t) / (k * L * (k * L * t + h))
        G[~pos] = d / k + 1 / h
    else:  # backside
        G[pos] = 1.0 / (k * L * np.sinh(L * d) + h * np.cosh(L * d))
        G[~pos] = 1 / h
    return G
 
 
def analytical_temperature(xs, ys, n_terms=SERIES_TERMS, face="device"):
    """Analytical T - T_amb at the points (xs[i], ys[i]) on the chosen face."""
    m = np.arange(n_terms)
    coeff = power_coefficients(n_terms) * modal_response(n_terms, face)   # (m, n)
    cos_x = np.cos(np.outer(xs, m) * np.pi / a)                          # (points, m)
    cos_y = np.cos(np.outer(ys, m) * np.pi / b)                          # (points, n)
    return T_AMBIENT + np.einsum("pm,mn,pn->p", cos_x, coeff, cos_y)
 
 
# ------------------------------------------------------------------
# 3. FEM solve on one mesh
# ------------------------------------------------------------------
def solve_fem(cells_x, cells_y, cells_z):
    t_start = time.perf_counter()
    domain = mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([a, b, d])],
        [cells_x, cells_y, cells_z],
        cell_type=mesh.CellType.tetrahedron,
    )
    tdim, fdim = 3, 2
    domain.topology.create_connectivity(fdim, tdim)
 
    face_locators = [
        (DEVICE_FACE,   lambda x: np.isclose(x[2], 0.0)),
        (BACKSIDE_FACE, lambda x: np.isclose(x[2], d)),
        (SIDE_X_MIN,    lambda x: np.isclose(x[0], 0.0)),
        (SIDE_X_MAX,    lambda x: np.isclose(x[0], a)),
        (SIDE_Y_MIN,    lambda x: np.isclose(x[1], 0.0)),
        (SIDE_Y_MAX,    lambda x: np.isclose(x[1], b)),
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
 
    x = ufl.SpatialCoordinate(domain)
    power_density = fem.Constant(domain, default_scalar_type(BACKGROUND_POWER_DENSITY))
    for (x0, x1, y0, y1, p) in POWER_BLOCKS.values():
        inside = ufl.And(ufl.And(ufl.ge(x[0], x0), ufl.le(x[0], x1)),
                         ufl.And(ufl.ge(x[1], y0), ufl.le(x[1], y1)))
        power_density = power_density + ufl.conditional(inside, p - BACKGROUND_POWER_DENSITY, 0.0)
 
    V = fem.functionspace(domain, ("Lagrange", 1))
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = fem.Constant(domain, default_scalar_type(K_SILICON))
    h = fem.Constant(domain, default_scalar_type(H_HEATSINK))
    T_amb = fem.Constant(domain, default_scalar_type(T_AMBIENT))
    F = (k * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
         + h * (T - T_amb) * v * ds(BACKSIDE_FACE)
         - power_density * v * ds(DEVICE_FACE))
 
    problem = LinearProblem(
        ufl.lhs(F), ufl.rhs(F), bcs=[],
        petsc_options_prefix=f"chip3d_powermap_{cells_x}_{cells_z}_",
        petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-12},
    )
    T_solution = problem.solve()
    runtime = time.perf_counter() - t_start
 
    def integrate(expr):
        return comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)
 
    total_power = integrate(power_density * ds(DEVICE_FACE))
    heat_out_backside = integrate(h * (T_solution - T_amb) * ds(BACKSIDE_FACE))
 
    # probe value
    tree = geometry.bb_tree(domain, tdim)
    p = PROBE_POINT.reshape(1, 3)
    cell = geometry.compute_colliding_cells(domain, geometry.compute_collisions_points(tree, p), p).links(0)[0]
    T_probe = T_solution.eval(p, np.array([cell], dtype=np.int32))[0]
 
    coords = V.tabulate_dof_coordinates()
    return {
        "T": T_solution.x.array.real.copy(),
        "coords": coords,
        "T_probe": T_probe,
        "total_power": total_power,
        "heat_out_backside": heat_out_backside,
        "cells": domain.topology.index_map(tdim).size_global,
        "unknowns": V.dofmap.index_map.size_global,
        "runtime": runtime,
    }
 
 
# ------------------------------------------------------------------
# 4. Run
# ------------------------------------------------------------------
if comm.rank == 0:
    # 4a. Analytical: how many series terms are enough?
    print("Series truncation check (analytical T at the CORE_LEFT centre, DEVICE_FACE)")
    for n_terms in SERIES_TERMS_CHECK:
        val = analytical_temperature(np.array([PROBE_POINT[0]]), np.array([PROBE_POINT[1]]), n_terms)[0]
        print(f"  {n_terms:4d} x {n_terms:<4d} terms : {val:.6f} degC")
    print()
 
T_probe_exact = analytical_temperature(np.array([PROBE_POINT[0]]), np.array([PROBE_POINT[1]]))[0]
 
rows = []
plot_data = None
for cells in MESH_LIST:
    if comm.rank == 0:
        print(f"solving FEM on {cells} ...", flush=True)
    res = solve_fem(*cells)
 
    on_device = np.isclose(res["coords"][:, 2], 0.0)
    xy = res["coords"][on_device, :2]
    T_fem = res["T"][on_device]
    T_exact = analytical_temperature(xy[:, 0], xy[:, 1])
    err = T_fem - T_exact
    rise_max = T_exact.max() - T_AMBIENT
 
    rows.append({
        "mesh": f"{cells[0]}x{cells[1]}x{cells[2]}",
        "h_xy_mm": round(a / cells[0] / MM, 4),
        "unknowns": res["unknowns"],
        "runtime_s": round(res["runtime"], 2),
        "T_probe_fem": round(res["T_probe"], 4),
        "T_probe_exact": round(T_probe_exact, 4),
        "probe_error": float(f"{res['T_probe'] - T_probe_exact:.3e}"),
        "max_abs_error_device_face": float(f"{np.abs(err).max():.3e}"),
        "max_error_pct_of_peak_rise": round(np.abs(err).max() / rise_max * 100, 4),
        "energy_balance_error": float(f"{abs(res['heat_out_backside'] - res['total_power']) / res['total_power']:.1e}"),
    })
    if cells == PLOT_MESH:
        plot_data = (xy, T_fem, T_exact, err, res)
 
if comm.rank == 0:
    # 4b. Error table
    print()
    print(f"{'mesh':>10s} {'h[mm]':>6s} {'unknowns':>9s} {'time[s]':>7s} {'T_probe FEM':>12s} {'exact':>9s} "
          f"{'probe err':>10s} {'max|err|':>9s} {'% of peak':>9s} {'order':>6s}")
    for i, r in enumerate(rows):
        order = (np.log2(rows[i - 1]["max_abs_error_device_face"] / r["max_abs_error_device_face"])
                 if i > 0 else np.nan)
        print(f"{r['mesh']:>10s} {r['h_xy_mm']:6.3f} {r['unknowns']:9d} {r['runtime_s']:7.2f} "
              f"{r['T_probe_fem']:12.4f} {r['T_probe_exact']:9.4f} {r['probe_error']:10.2e} "
              f"{r['max_abs_error_device_face']:9.2e} {r['max_error_pct_of_peak_rise']:9.4f} {order:6.2f}")
 
    with open("power_map_error_vs_mesh.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
 
    # 4c. Summary for the plotted mesh
    xy, T_fem, T_exact, err, res = plot_data
    chip_area = a * b
    expected_avg = T_AMBIENT + res["total_power"] * (d / (K_SILICON * chip_area) + 1 / (H_HEATSINK * chip_area))
    i_fem, i_exact = np.argmax(T_fem), np.argmax(T_exact)
    print()
    print(f"Mesh {PLOT_MESH}:")
    print(f"  total power                 {res['total_power']:.3f} W")
    print(f"  heat out through backside   {res['heat_out_backside']:.3f} W  (Robin h(T-Ta))")
    print(f"  expected average rise       {expected_avg - T_AMBIENT:.3f} degC  (P (d/kA + 1/hA))")
    print(f"  hottest node, FEM           {T_fem[i_fem]:.3f} degC at (x, y) = "
          f"({xy[i_fem, 0] / MM:.2f}, {xy[i_fem, 1] / MM:.2f}) mm")
    print(f"  hottest node, analytical    {T_exact[i_exact]:.3f} degC at (x, y) = "
          f"({xy[i_exact, 0] / MM:.2f}, {xy[i_exact, 1] / MM:.2f}) mm")
    i_err = np.argmax(np.abs(err))
    print(f"  largest |FEM - analytical|  {abs(err[i_err]):.2e} degC at (x, y) = "
          f"({xy[i_err, 0] / MM:.2f}, {xy[i_err, 1] / MM:.2f}) mm")
 
    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    X, Y = xy[:, 0] / MM, xy[:, 1] / MM
 
    # power map on a regular grid (for display only)
    gx = np.linspace(0, a, 201)
    GX, GY = np.meshgrid(gx, gx, indexing="ij")
    Pd_grid = np.full_like(GX, BACKGROUND_POWER_DENSITY)
    for (x0, x1, y0, y1, p) in POWER_BLOCKS.values():
        Pd_grid[(GX >= x0) & (GX <= x1) & (GY >= y0) & (GY <= y1)] = p
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    pc = ax.pcolormesh(GX / MM, GY / MM, Pd_grid / W_PER_CM2, cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=ax, label="power density (W/cm²)")
    for name, (x0, x1, y0, y1, _) in POWER_BLOCKS.items():
        ax.text((x0 + x1) / 2 / MM, (y0 + y1) / 2 / MM, name, color="w", ha="center", va="center", fontsize=9)
    ax.set_title(f"Power map   (total {res['total_power']:.1f} W)")
 
    t_levels = np.linspace(min(T_fem.min(), T_exact.min()), max(T_fem.max(), T_exact.max()), 41)
    for ax, data, title in [(axes[0, 1], T_fem, f"FEM {PLOT_MESH}"),
                            (axes[1, 0], T_exact, f"Analytical ({SERIES_TERMS}x{SERIES_TERMS} terms)")]:
        cf = ax.tricontourf(X, Y, data, levels=t_levels, cmap="hot")
        fig.colorbar(cf, ax=ax, label="T - T_ambient (°C)")
        ax.set_title(f"{title}   max = {data.max():.2f} °C")
 
    ax = axes[1, 1]
    lim = np.abs(err).max()
    cf = ax.tricontourf(X, Y, err, levels=np.linspace(-lim, lim, 41), cmap="coolwarm")
    fig.colorbar(cf, ax=ax, label="FEM - analytical (°C)")
    ax.set_title(f"Error   max |err| = {lim:.3f} °C")
 
    for ax in axes.flat:
        for (x0, x1, y0, y1, _) in POWER_BLOCKS.values():
            ax.add_patch(plt.Rectangle((x0 / MM, y0 / MM), (x1 - x0) / MM, (y1 - y0) / MM,
                                       fill=False, ec="cyan", lw=0.8, ls="--"))
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    fig.suptitle("DEVICE_FACE temperature: FEM vs. analytical cosine series")
    fig.tight_layout()
    fig.savefig("power_map_fem_vs_analytical.png", dpi=200)
 
    # error vs mesh size
    h_vals = np.array([r["h_xy_mm"] for r in rows])
    max_err = np.array([r["max_abs_error_device_face"] for r in rows])
    probe_err = np.abs([r["probe_error"] for r in rows])
    plt.figure(figsize=(7, 5))
    plt.loglog(h_vals, max_err, "o-", label="max |FEM - analytical| on DEVICE_FACE")
    plt.loglog(h_vals, probe_err, "s-", label="|error| at CORE_LEFT centre")
    plt.loglog(h_vals, max_err[0] * (h_vals / h_vals[0]) ** 2, "k--", label="slope 2 (reference)")
    plt.loglog(h_vals, max_err[0] * (h_vals / h_vals[0]) ** 1, "k:", label="slope 1 (reference)")
    plt.xlabel("x-y cell size h (mm)")
    plt.ylabel("error (°C)")
    plt.title("FEM error against the analytical solution")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("power_map_error_vs_mesh.png", dpi=200)
    plt.show()