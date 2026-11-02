from pathlib import Path
import csv
import time

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem

# All meshes solve exactly the same physical model.
XC, YC = 0.68, 0.38
Q0 = 100.0
SIGMA = 0.08
K_THERMAL = 1.0
T_BOUNDARY = 300.0

UNIFORM_NS = [20, 40, 80, 160]
LOCAL_BASE_N = 20
LOCAL_RADII = [0.22, 0.12, 0.06]
OUTDIR = Path(__file__).resolve().parent / "benchmark_output"


def uniform_mesh(n):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [n, n],
        cell_type=mesh.CellType.triangle,
    )


def local_mesh():
    domain = uniform_mesh(LOCAL_BASE_N)
    for radius in LOCAL_RADII:
        domain.topology.create_entities(1)
        domain.topology.create_connectivity(2, 1)
        cell_map = domain.topology.index_map(2)
        cells = np.arange(cell_map.size_local, dtype=np.int32)
        midpoints = mesh.compute_midpoints(domain, 2, cells)
        distance_sq = (midpoints[:, 0] - XC) ** 2 + (midpoints[:, 1] - YC) ** 2
        marked = cells[distance_sq < radius ** 2]
        if marked.size == 0:
            continue
        edges = mesh.compute_incident_entities(domain.topology, marked, 2, 1)
        edges = np.unique(edges).astype(np.int32)
        domain, _, _ = mesh.refine(domain, edges)
    return domain


def solve(domain):
    start = time.perf_counter()
    V = fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    q = Q0 * ufl.exp(-((x[0] - XC) ** 2 + (x[1] - YC) ** 2) / (2 * SIGMA ** 2))
    a = K_THERMAL * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = q * v * ufl.dx

    # DOLFINx requires facet entities and facet -> cell connectivity
    # before exterior_facet_indices can identify boundary facets.
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)
    domain.topology.create_connectivity(fdim, tdim)
    facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    boundary_values = fem.Function(V)
    boundary_values.x.array[:] = T_BOUNDARY
    bc = fem.dirichletbc(boundary_values, dofs)

    # Robust single-process reference solver; for large-scale problems
    # an iterative solver may be preferable.
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix="fair_thermal_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    uh = problem.solve()
    uh.x.scatter_forward()
    elapsed = time.perf_counter() - start

    peak = float(MPI.COMM_WORLD.allreduce(np.max(uh.x.array), op=MPI.MAX))
    cells_count = int(MPI.COMM_WORLD.allreduce(
        domain.topology.index_map(tdim).size_local, op=MPI.SUM))
    dofs_count = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
    return peak, cells_count, dofs_count, elapsed

def main():
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run using /usr/bin/python3 (single MPI process), not mpirun")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("Same physics: center=(0.68,0.38), q0=100, sigma=0.08, k=1, boundary=300 K", flush=True)

    for n in UNIFORM_NS:
        t0 = time.perf_counter()
        domain = uniform_mesh(n)
        mesh_time = time.perf_counter() - t0
        peak, cells, dofs, fem_time = solve(domain)
        rows.append({"mesh": f"Uniform N={n}", "elements": cells, "dofs": dofs,
                     "peak_K": peak, "mesh_s": mesh_time, "fem_s": fem_time})
        print(f"Uniform N={n:<3}: {cells:>7} cells, {dofs:>6} DOFs, peak={peak:.9f} K", flush=True)

    t0 = time.perf_counter()
    local_domain = local_mesh()
    mesh_time = time.perf_counter() - t0
    peak, cells, dofs, fem_time = solve(local_domain)
    rows.append({"mesh": "Local r=0.22/0.12/0.06", "elements": cells, "dofs": dofs,
                 "peak_K": peak, "mesh_s": mesh_time, "fem_s": fem_time})
    print(f"Local:         {cells:>7} cells, {dofs:>6} DOFs, peak={peak:.9f} K", flush=True)

    reference = rows[len(UNIFORM_NS) - 1]["peak_K"]
    for row in rows:
        row["peak_error_K"] = abs(row["peak_K"] - reference)
        row["total_s"] = row["mesh_s"] + row["fem_s"]

    print(f"\nNumerical reference: Uniform N={UNIFORM_NS[-1]} peak={reference:.9f} K")
    print(f"{'Mesh':26} {'Cells':>8} {'DOFs':>8} {'Peak K':>14} {'Error K':>13} {'Mesh s':>10} {'FEM s':>10} {'Total s':>10}")
    for r in rows:
        print(f"{r['mesh']:26} {r['elements']:8d} {r['dofs']:8d} {r['peak_K']:14.8f} {r['peak_error_K']:13.7g} {r['mesh_s']:10.4f} {r['fem_s']:10.4f} {r['total_s']:10.4f}")

    output = OUTDIR / "fair_benchmark.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    uniforms, local = rows[:len(UNIFORM_NS)], rows[-1]
    # Reference point has zero error, so it cannot be displayed on a log axis.
    plot_uniforms = uniforms[:-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog([r["elements"] for r in plot_uniforms],
              [r["peak_error_K"] for r in plot_uniforms], "o-", label="Uniform")
    if local["peak_error_K"] > 0:
        ax.loglog(local["elements"], local["peak_error_K"], "*", markersize=15,
                  label="Local refinement")
    ax.set(xlabel="Number of triangular elements", ylabel="Peak temperature deviation (K)",
           title="Fair comparison: same off-center heat source")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "accuracy_vs_elements.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(rows))
    mesh_times = [r["mesh_s"] for r in rows]
    fem_times = [r["fem_s"] for r in rows]
    ax.bar(xs, mesh_times, label="Mesh creation/refinement")
    ax.bar(xs, fem_times, bottom=mesh_times, label="FEM setup + assembly + solve")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["mesh"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.legend()
    ax.set_title("One-run timings (first-run compilation may affect time)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "runtime_breakdown.png", dpi=200)
    plt.close(fig)

    print("\nSaved:", output)
    print("Figures:", OUTDIR / "accuracy_vs_elements.png", "and", OUTDIR / "runtime_breakdown.png")
    print("NOTE: N=160 is a numerical reference, not an exact solution.")
    print("NOTE: Timings include assembly/JIT effects; repeat for robust performance conclusions.")


if __name__ == "__main__":
    main()


